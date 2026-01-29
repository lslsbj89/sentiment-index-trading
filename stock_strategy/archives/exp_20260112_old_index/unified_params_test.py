"""
统一参数测试 - 对比原最优参数 vs 新统一参数
卖出条件: (情绪 > X AND 价格 < MA50) OR (情绪 > 30)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import psycopg2

# 配置中文字体
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams['font.family'] = ['Heiti TC', 'STHeiti', 'PingFang HK', 'Songti SC']
plt.rcParams['axes.unicode_minus'] = False

from data_loader import DataLoader

db_config = {
    "host": "localhost",
    "port": 5432,
    "database": "crypto_fear_greed_2",
    "user": "sc2025",
    "password": ""
}

INITIAL_CAPITAL = 100000
COMMISSION_RATE = 0.001
SLIPPAGE_RATE = 0.001
POSITION_PCT = 0.8
MA_PERIOD = 50

MAG7_STOCKS = ['NVDA', 'TSLA', 'GOOGL', 'META', 'MSFT', 'AAPL', 'AMZN']
TEST_YEARS = [2021, 2022, 2023, 2024, 2025]

# 原最优参数 (2021-2025实验结果)
ORIGINAL_CONFIG = {
    'NVDA': {'buy_th': -10, 'sell_mode': 'and_ma', 'sell_th': 10},
    'TSLA': {'buy_th': -10, 'sell_mode': 'and_ma', 'sell_th': 10},
    'GOOGL': {'buy_th': -5, 'sell_mode': 'threshold', 'sell_th': 30},
    'META': {'buy_th': -10, 'sell_mode': 'and_ma', 'sell_th': 5},
    'MSFT': {'buy_th': -10, 'sell_mode': 'threshold', 'sell_th': 30},
    'AAPL': {'buy_th': -10, 'sell_mode': 'threshold', 'sell_th': 30},
    'AMZN': {'buy_th': -10, 'sell_mode': 'and_ma', 'sell_th': 5},
}

# 新统一参数: (情绪 > X AND 价格 < MA50) OR (情绪 > 40)
OR_THRESHOLD = 40  # 可调整: 30, 40, 50

UNIFIED_CONFIG = {
    'NVDA': {'buy_th': -10, 'sell_mode': 'and_or', 'sell_th': 10},
    'TSLA': {'buy_th': -10, 'sell_mode': 'and_or', 'sell_th': 10},
    'GOOGL': {'buy_th': -5, 'sell_mode': 'and_or', 'sell_th': 10},
    'META': {'buy_th': -10, 'sell_mode': 'and_or', 'sell_th': 5},
    'MSFT': {'buy_th': -10, 'sell_mode': 'and_or', 'sell_th': 10},
    'AAPL': {'buy_th': -10, 'sell_mode': 'and_or', 'sell_th': 10},
    'AMZN': {'buy_th': -10, 'sell_mode': 'and_or', 'sell_th': 5},
}


def load_sentiment_data(symbol):
    conn = psycopg2.connect(**db_config)
    query = f"""
        SELECT date, smoothed_index
        FROM fear_greed_index
        WHERE symbol = '{symbol}'
        ORDER BY date
    """
    df = pd.read_sql(query, conn)
    conn.close()
    if len(df) == 0:
        return None
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.set_index('date')
    return df


def run_yearly_backtest(stock_data, sentiment_data, config, year):
    """运行单年度回测"""
    buy_th = config['buy_th']
    sell_mode = config['sell_mode']
    sell_th = config['sell_th']

    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    lookback_start = pd.to_datetime(start_date, utc=True) - pd.Timedelta(days=MA_PERIOD * 2)
    end_dt = pd.to_datetime(end_date, utc=True)
    mask = (stock_data.index >= lookback_start) & (stock_data.index <= end_dt)
    price_data = stock_data[mask].copy()

    if len(price_data) < MA_PERIOD:
        return None

    price_data['MA'] = price_data['Close'].rolling(window=MA_PERIOD).mean()

    start_dt = pd.to_datetime(start_date, utc=True)
    test_mask = price_data.index >= start_dt
    price_data = price_data[test_mask].copy()

    if len(price_data) == 0 or sentiment_data is None:
        return None

    sent_data = sentiment_data.reindex(price_data.index)

    cash = INITIAL_CAPITAL
    shares = 0
    entry_price = 0
    entry_date = None
    trades = []
    portfolio_values = []
    daily_returns = []
    prev_value = INITIAL_CAPITAL

    for i, (date, row) in enumerate(price_data.iterrows()):
        price = row['Close']
        ma = row['MA']
        sentiment = sent_data.loc[date, 'smoothed_index'] if date in sent_data.index else None

        current_total = cash + shares * price

        if pd.isna(sentiment) or pd.isna(ma):
            portfolio_values.append(current_total)
            daily_returns.append((current_total - prev_value) / prev_value if prev_value > 0 else 0)
            prev_value = current_total
            continue

        # 卖出逻辑
        if shares > 0:
            sell_signal = False
            exit_reason = None

            if sell_mode == 'threshold':
                if sentiment > sell_th:
                    sell_signal = True
                    exit_reason = f'>{sell_th}'
            elif sell_mode == 'and_ma':
                if sentiment > sell_th and price < ma:
                    sell_signal = True
                    exit_reason = f'>{sell_th}&<MA'
            elif sell_mode == 'and_or':
                # 新条件: (情绪 > X AND 价格 < MA) OR (情绪 > OR_THRESHOLD)
                if sentiment > sell_th and price < ma:
                    sell_signal = True
                    exit_reason = f'>{sell_th}&<MA'
                elif sentiment > OR_THRESHOLD:
                    sell_signal = True
                    exit_reason = f'>{OR_THRESHOLD}'

            if sell_signal:
                sell_price = price * (1 - SLIPPAGE_RATE) * (1 - COMMISSION_RATE)
                proceeds = shares * sell_price
                cash += proceeds
                trade_return = (sell_price - entry_price) / entry_price
                trades.append({
                    'type': 'SELL',
                    'date': date,
                    'price': price,
                    'sentiment': sentiment,
                    'reason': exit_reason,
                    'trade_return': trade_return
                })
                shares = 0
                entry_price = 0

        # 买入逻辑 - 动态仓位
        elif shares == 0 and sentiment < buy_th:
            current_total_for_position = cash + shares * price
            position_value = current_total_for_position * POSITION_PCT

            buy_price = price * (1 + SLIPPAGE_RATE) * (1 + COMMISSION_RATE)
            shares = int(position_value / buy_price)

            if shares > 0:
                cost = shares * buy_price
                if cost <= cash:
                    cash -= cost
                    entry_price = buy_price
                    entry_date = date
                    trades.append({
                        'type': 'BUY',
                        'date': date,
                        'price': price,
                        'sentiment': sentiment,
                        'reason': f'<{buy_th}',
                        'trade_return': None
                    })
                else:
                    shares = 0

        current_total = cash + shares * price
        portfolio_values.append(current_total)
        daily_returns.append((current_total - prev_value) / prev_value if prev_value > 0 else 0)
        prev_value = current_total

    # 年末清仓
    if shares > 0:
        final_price = price_data.iloc[-1]['Close']
        sell_price = final_price * (1 - SLIPPAGE_RATE) * (1 - COMMISSION_RATE)
        cash += shares * sell_price
        trade_return = (sell_price - entry_price) / entry_price
        trades.append({
            'type': 'SELL',
            'date': price_data.index[-1],
            'price': final_price,
            'sentiment': sent_data.iloc[-1]['smoothed_index'] if len(sent_data) > 0 else None,
            'reason': 'year_end',
            'trade_return': trade_return
        })

    final_value = cash
    total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL

    # 最大回撤
    if len(portfolio_values) > 0:
        portfolio_series = pd.Series(portfolio_values, index=price_data.index)
        rolling_max = portfolio_series.expanding().max()
        drawdowns = (portfolio_series - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
    else:
        max_drawdown = 0

    # 夏普率
    daily_returns_arr = np.array(daily_returns)
    if len(daily_returns_arr) > 0 and np.std(daily_returns_arr) > 0:
        sharpe = np.mean(daily_returns_arr) / np.std(daily_returns_arr) * np.sqrt(252)
    else:
        sharpe = 0

    # 交易统计
    sell_trades = [t for t in trades if t['type'] == 'SELL']
    trade_count = len(sell_trades)
    if trade_count > 0:
        wins = sum(1 for t in sell_trades if t['trade_return'] and t['trade_return'] > 0)
        win_rate = wins / trade_count
    else:
        win_rate = 0

    return {
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'trade_count': trade_count,
        'win_rate': win_rate,
        'trades': trades
    }


def main():
    print("=" * 100)
    print("统一参数测试: 原最优参数 vs 新统一参数")
    print(f"新卖出条件: (情绪 > X AND 价格 < MA50) OR (情绪 > {OR_THRESHOLD})")
    print("=" * 100)

    loader = DataLoader(db_config)

    # 存储对比结果
    comparison_results = []

    for symbol in MAG7_STOCKS:
        print(f"\n{'='*80}")
        print(f"{symbol}")
        print("=" * 80)

        stock_data = loader.load_ohlcv(symbol, start_date="2019-01-01", end_date="2025-12-31")
        sentiment_data = load_sentiment_data(symbol)

        if stock_data is None or sentiment_data is None:
            print(f"  无数据")
            continue

        orig_config = ORIGINAL_CONFIG[symbol]
        unif_config = UNIFIED_CONFIG[symbol]

        # 原配置描述
        orig_desc = f"buy<{orig_config['buy_th']}, "
        if orig_config['sell_mode'] == 'threshold':
            orig_desc += f"sell>{orig_config['sell_th']}"
        else:
            orig_desc += f"sell>{orig_config['sell_th']}&<MA"

        # 新配置描述
        unif_desc = f"buy<{unif_config['buy_th']}, sell>(>{unif_config['sell_th']}&<MA) OR >{OR_THRESHOLD}"

        print(f"原参数: {orig_desc}")
        print(f"新参数: {unif_desc}")

        # 年度对比
        print(f"\n{'年份':<6} {'原-收益':>10} {'新-收益':>10} {'差异':>10} | {'原-夏普':>8} {'新-夏普':>8} | {'原-交易':>6} {'新-交易':>6}")
        print("-" * 85)

        orig_returns = []
        unif_returns = []
        orig_sharpes = []
        unif_sharpes = []

        for year in TEST_YEARS:
            orig_result = run_yearly_backtest(stock_data, sentiment_data, orig_config, year)
            unif_result = run_yearly_backtest(stock_data, sentiment_data, unif_config, year)

            if orig_result and unif_result:
                orig_ret = orig_result['total_return']
                unif_ret = unif_result['total_return']
                diff = unif_ret - orig_ret

                orig_returns.append(orig_ret)
                unif_returns.append(unif_ret)
                orig_sharpes.append(orig_result['sharpe'])
                unif_sharpes.append(unif_result['sharpe'])

                better = "新" if diff > 0.01 else ("原" if diff < -0.01 else "平")

                print(f"{year:<6} {orig_ret:>+10.2%} {unif_ret:>+10.2%} {diff:>+10.2%} | "
                      f"{orig_result['sharpe']:>8.2f} {unif_result['sharpe']:>8.2f} | "
                      f"{orig_result['trade_count']:>6} {unif_result['trade_count']:>6}  [{better}]")

        # 汇总
        if orig_returns and unif_returns:
            orig_cum = np.prod([1 + r for r in orig_returns]) - 1
            unif_cum = np.prod([1 + r for r in unif_returns]) - 1
            orig_avg_sharpe = np.mean(orig_sharpes)
            unif_avg_sharpe = np.mean(unif_sharpes)

            print("-" * 85)
            print(f"{'5年累计':<6} {orig_cum:>+10.2%} {unif_cum:>+10.2%} {unif_cum - orig_cum:>+10.2%} | "
                  f"{orig_avg_sharpe:>8.2f} {unif_avg_sharpe:>8.2f}")

            comparison_results.append({
                'symbol': symbol,
                'orig_config': orig_desc,
                'unif_config': unif_desc,
                'orig_cumulative': orig_cum,
                'unif_cumulative': unif_cum,
                'diff': unif_cum - orig_cum,
                'orig_sharpe': orig_avg_sharpe,
                'unif_sharpe': unif_avg_sharpe,
                'better': '新参数' if unif_cum > orig_cum else '原参数'
            })

    loader.close()

    # 总结
    print("\n" + "=" * 100)
    print("汇总对比")
    print("=" * 100)

    print(f"\n{'股票':<8} {'原参数5年累计':>15} {'新参数5年累计':>15} {'差异':>12} {'更优':>10}")
    print("-" * 70)

    orig_total = 0
    unif_total = 0
    new_wins = 0

    for r in comparison_results:
        print(f"{r['symbol']:<8} {r['orig_cumulative']:>+15.2%} {r['unif_cumulative']:>+15.2%} "
              f"{r['diff']:>+12.2%} {r['better']:>10}")
        orig_total += r['orig_cumulative']
        unif_total += r['unif_cumulative']
        if r['unif_cumulative'] > r['orig_cumulative']:
            new_wins += 1

    print("-" * 70)
    print(f"{'平均':<8} {orig_total/len(comparison_results):>+15.2%} {unif_total/len(comparison_results):>+15.2%} "
          f"{(unif_total-orig_total)/len(comparison_results):>+12.2%}")
    print(f"\n新参数胜出: {new_wins}/{len(comparison_results)} 只股票")

    # 新参数配置表
    print("\n" + "=" * 100)
    print("新统一参数配置表")
    print("=" * 100)
    print(f"""
| 股票  | 买入条件 | 卖出条件                      |
|-------|----------|-------------------------------|
| NVDA  | <-10     | (>10 AND <MA50) OR >{OR_THRESHOLD}        |
| TSLA  | <-10     | (>10 AND <MA50) OR >{OR_THRESHOLD}        |
| GOOGL | <-5      | (>10 AND <MA50) OR >{OR_THRESHOLD}        |
| META  | <-10     | (>5 AND <MA50) OR >{OR_THRESHOLD}         |
| MSFT  | <-10     | (>10 AND <MA50) OR >{OR_THRESHOLD}        |
| AAPL  | <-10     | (>10 AND <MA50) OR >{OR_THRESHOLD}        |
| AMZN  | <-10     | (>5 AND <MA50) OR >{OR_THRESHOLD}         |

特点:
- 买入条件: 大部分 <-10，GOOGL 放宽至 <-5
- 卖出条件: AND止损 + 阈值兜底(>{OR_THRESHOLD})
- 全部动态仓位 80%
""")

    print("\n完成!")


if __name__ == "__main__":
    main()
