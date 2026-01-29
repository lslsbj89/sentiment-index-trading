"""
测试新股票的情绪策略
UBER, HOOD, COIN, BABA
测试所有买入/卖出条件组合，找出最优配置
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

# 新股票列表
NEW_STOCKS = ['UBER', 'HOOD', 'COIN', 'BABA']

# 测试配置组合
TEST_CONFIGS = [
    {'buy_th': -10, 'sell_mode': 'threshold', 'sell_th': 30, 'position_mode': 'fixed', 'name': 'buy<-10, sell>30, 固定'},
    {'buy_th': -10, 'sell_mode': 'and_ma', 'sell_th': 10, 'position_mode': 'dynamic', 'name': 'buy<-10, sell>10&MA, 动态'},
    {'buy_th': -10, 'sell_mode': 'and_ma', 'sell_th': 5, 'position_mode': 'dynamic', 'name': 'buy<-10, sell>5&MA, 动态'},
    {'buy_th': -5, 'sell_mode': 'threshold', 'sell_th': 30, 'position_mode': 'fixed', 'name': 'buy<-5, sell>30, 固定'},
    {'buy_th': -5, 'sell_mode': 'and_ma', 'sell_th': 10, 'position_mode': 'dynamic', 'name': 'buy<-5, sell>10&MA, 动态'},
    {'buy_th': -5, 'sell_mode': 'and_ma', 'sell_th': 5, 'position_mode': 'dynamic', 'name': 'buy<-5, sell>5&MA, 动态'},
]


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


def run_backtest(stock_data, sentiment_data, config):
    """运行回测"""
    buy_th = config['buy_th']
    sell_mode = config['sell_mode']
    sell_th = config['sell_th']
    use_dynamic = config['position_mode'] == 'dynamic'

    start_date = "2021-01-01"
    end_date = "2025-12-31"

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
    entry_sentiment = None
    trades = []
    portfolio_values = []
    daily_returns = []
    prev_value = INITIAL_CAPITAL
    fixed_position_value = INITIAL_CAPITAL * POSITION_PCT

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
                    exit_reason = f'sentiment>{sell_th}'
            elif sell_mode == 'and_ma':
                if sentiment > sell_th and price < ma:
                    sell_signal = True
                    exit_reason = f'>{sell_th}&<MA'

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
                    'ma': ma,
                    'reason': exit_reason,
                    'shares': shares,
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'trade_return': trade_return
                })
                shares = 0
                entry_price = 0

        # 买入逻辑
        elif shares == 0 and sentiment < buy_th:
            if use_dynamic:
                position_value = cash * POSITION_PCT
            else:
                position_value = min(fixed_position_value, cash)

            buy_price = price * (1 + SLIPPAGE_RATE) * (1 + COMMISSION_RATE)
            shares = int(position_value / buy_price)

            if shares > 0:
                cost = shares * buy_price
                if cost <= cash:
                    cash -= cost
                    entry_price = buy_price
                    entry_date = date
                    entry_sentiment = sentiment
                    trades.append({
                        'type': 'BUY',
                        'date': date,
                        'price': price,
                        'sentiment': sentiment,
                        'ma': ma,
                        'reason': f'sentiment<{buy_th}',
                        'shares': shares,
                        'entry_date': None,
                        'entry_price': None,
                        'trade_return': None
                    })
                else:
                    shares = 0

        current_total = cash + shares * price
        portfolio_values.append(current_total)
        daily_returns.append((current_total - prev_value) / prev_value if prev_value > 0 else 0)
        prev_value = current_total

    # 最终清仓
    if shares > 0:
        final_price = price_data.iloc[-1]['Close']
        final_sentiment = sent_data.iloc[-1]['smoothed_index'] if len(sent_data) > 0 else None
        final_ma = price_data.iloc[-1]['MA']
        sell_price = final_price * (1 - SLIPPAGE_RATE) * (1 - COMMISSION_RATE)
        cash += shares * sell_price
        trade_return = (sell_price - entry_price) / entry_price
        trades.append({
            'type': 'SELL',
            'date': price_data.index[-1],
            'price': final_price,
            'sentiment': final_sentiment,
            'ma': final_ma,
            'reason': 'end',
            'shares': shares,
            'entry_date': entry_date,
            'entry_price': entry_price,
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
        portfolio_series = pd.Series()

    # 年化收益
    years = 5
    annualized = (1 + total_return) ** (1/years) - 1 if total_return > -1 else -1

    # 夏普率
    daily_returns_arr = np.array(daily_returns)
    if len(daily_returns_arr) > 0 and np.std(daily_returns_arr) > 0:
        sharpe = np.mean(daily_returns_arr) / np.std(daily_returns_arr) * np.sqrt(252)
    else:
        sharpe = 0

    # 胜率
    sell_trades = [t for t in trades if t['type'] == 'SELL']
    if len(sell_trades) > 0:
        wins = sum(1 for t in sell_trades if t['trade_return'] and t['trade_return'] > 0)
        win_rate = wins / len(sell_trades)
    else:
        win_rate = 0

    return {
        'total_return': total_return,
        'annualized': annualized,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'trades': trades,
        'trade_count': len(sell_trades),
        'win_rate': win_rate,
        'final_value': final_value,
        'portfolio_series': portfolio_series,
        'price_data': price_data
    }


def analyze_sentiment(symbol, sentiment_data):
    """分析情绪指数特征"""
    start_dt = pd.to_datetime('2021-01-01', utc=True)
    end_dt = pd.to_datetime('2025-12-31', utc=True)
    mask = (sentiment_data.index >= start_dt) & (sentiment_data.index <= end_dt)
    sent = sentiment_data[mask]['smoothed_index']

    if len(sent) == 0:
        return None

    return {
        'min': sent.min(),
        'max': sent.max(),
        'mean': sent.mean(),
        'std': sent.std(),
        'days_below_minus10': (sent < -10).sum(),
        'days_below_minus5': (sent < -5).sum(),
        'days_above_30': (sent > 30).sum(),
        'days_above_10': (sent > 10).sum(),
        'days_above_5': (sent > 5).sum(),
    }


def main():
    print("=" * 100)
    print("新股票情绪策略测试 - UBER, HOOD, COIN, BABA")
    print("=" * 100)

    loader = DataLoader(db_config)

    # 创建输出目录
    output_dir = 'new_stocks_results'
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    for symbol in NEW_STOCKS:
        print(f"\n{'='*80}")
        print(f"测试 {symbol}")
        print("=" * 80)

        stock_data = loader.load_ohlcv(symbol, start_date="2014-01-01")
        sentiment_data = load_sentiment_data(symbol)

        if stock_data is None or len(stock_data) == 0:
            print(f"  ⚠ {symbol} 无价格数据")
            continue

        if sentiment_data is None or len(sentiment_data) == 0:
            print(f"  ⚠ {symbol} 无情绪数据")
            continue

        # 分析情绪特征
        sent_stats = analyze_sentiment(symbol, sentiment_data)
        if sent_stats:
            print(f"\n情绪指数统计 (2021-2025):")
            print(f"  范围: {sent_stats['min']:.1f} ~ {sent_stats['max']:.1f}")
            print(f"  均值: {sent_stats['mean']:.1f}, 标准差: {sent_stats['std']:.1f}")
            print(f"  情绪<-10 天数: {sent_stats['days_below_minus10']}")
            print(f"  情绪<-5 天数: {sent_stats['days_below_minus5']}")
            print(f"  情绪>30 天数: {sent_stats['days_above_30']}")
            print(f"  情绪>10 天数: {sent_stats['days_above_10']}")

        # 测试所有配置
        print(f"\n测试6种配置组合:")
        print("-" * 90)
        print(f"{'配置':<30} {'5年收益':>12} {'年化':>10} {'回撤':>10} {'夏普':>8} {'交易':>6} {'胜率':>8}")
        print("-" * 90)

        symbol_results = []
        for config in TEST_CONFIGS:
            result = run_backtest(stock_data, sentiment_data, config)
            if result:
                symbol_results.append({
                    'symbol': symbol,
                    'config': config,
                    'result': result
                })
                print(f"{config['name']:<30} {result['total_return']:>12.2%} {result['annualized']:>10.2%} "
                      f"{result['max_drawdown']:>10.2%} {result['sharpe']:>8.2f} {result['trade_count']:>6} "
                      f"{result['win_rate']:>8.0%}")
            else:
                print(f"{config['name']:<30} {'无数据':>12}")

        # 找出最优配置
        if symbol_results:
            best = max(symbol_results, key=lambda x: x['result']['sharpe'])
            print("-" * 90)
            print(f"✓ {symbol} 最优配置: {best['config']['name']}")
            print(f"  收益: {best['result']['total_return']:.2%}, 夏普: {best['result']['sharpe']:.2f}")

            all_results.append(best)

    loader.close()

    # 汇总结果
    print("\n" + "=" * 100)
    print("新股票最优配置汇总")
    print("=" * 100)

    if all_results:
        print(f"\n{'股票':<8} {'最优配置':<30} {'5年收益':>12} {'年化':>10} {'回撤':>10} {'夏普':>8}")
        print("-" * 85)

        for r in sorted(all_results, key=lambda x: x['result']['total_return'], reverse=True):
            print(f"{r['symbol']:<8} {r['config']['name']:<30} {r['result']['total_return']:>12.2%} "
                  f"{r['result']['annualized']:>10.2%} {r['result']['max_drawdown']:>10.2%} "
                  f"{r['result']['sharpe']:>8.2f}")

        # 保存结果
        summary_data = []
        for r in all_results:
            summary_data.append({
                'symbol': r['symbol'],
                'buy_threshold': r['config']['buy_th'],
                'sell_mode': r['config']['sell_mode'],
                'sell_threshold': r['config']['sell_th'],
                'position_mode': r['config']['position_mode'],
                'total_return': r['result']['total_return'],
                'annualized': r['result']['annualized'],
                'max_drawdown': r['result']['max_drawdown'],
                'sharpe': r['result']['sharpe'],
                'trade_count': r['result']['trade_count'],
                'win_rate': r['result']['win_rate'],
                'final_value': r['result']['final_value']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, 'optimal_configs.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✓ 最优配置已保存: {summary_path}")

    print("\n" + "=" * 100)
    print("完成!")
    print("=" * 100)


if __name__ == "__main__":
    main()
