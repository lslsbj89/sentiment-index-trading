"""
年度独立回测
每年独立计算，年初重置资金，年末清仓
显示详细的买入卖出时间和价格
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import psycopg2

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

# 最优配置
OPTIMAL_CONFIG = {
    'NVDA': {'buy_th': -10, 'sell_mode': 'and_ma', 'sell_th': 10},
    'TSLA': {'buy_th': -10, 'sell_mode': 'and_ma', 'sell_th': 10},
    'GOOGL': {'buy_th': -5, 'sell_mode': 'threshold', 'sell_th': 30},
    'META': {'buy_th': -10, 'sell_mode': 'and_ma', 'sell_th': 5},
    'MSFT': {'buy_th': -10, 'sell_mode': 'threshold', 'sell_th': 30},
    'AAPL': {'buy_th': -10, 'sell_mode': 'threshold', 'sell_th': 30},
    'AMZN': {'buy_th': -10, 'sell_mode': 'and_ma', 'sell_th': 5},
}

TEST_YEARS = [2021, 2022, 2023, 2024, 2025]


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
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.set_index('date')
    return df


def run_yearly_backtest(stock_data, sentiment_data, symbol, year):
    """单年度回测，返回详细交易信息"""
    config = OPTIMAL_CONFIG[symbol]
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

    if len(price_data) == 0:
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
                cash += shares * sell_price
                trades.append({
                    'type': 'SELL',
                    'date': date,
                    'price': price,
                    'net_price': sell_price,
                    'sentiment': sentiment,
                    'ma': ma,
                    'reason': exit_reason,
                    'shares': shares,
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'entry_sentiment': entry_sentiment,
                    'trade_return': (sell_price - entry_price) / entry_price
                })
                shares = 0
                entry_price = 0

        # 买入逻辑
        elif shares == 0 and sentiment < buy_th:
            position_value = cash * POSITION_PCT
            buy_price = price * (1 + SLIPPAGE_RATE) * (1 + COMMISSION_RATE)
            shares = int(position_value / buy_price)
            if shares > 0:
                cash -= shares * buy_price
                entry_price = buy_price
                entry_date = date
                entry_sentiment = sentiment
                trades.append({
                    'type': 'BUY',
                    'date': date,
                    'price': price,
                    'net_price': buy_price,
                    'sentiment': sentiment,
                    'ma': ma,
                    'reason': f'sentiment<{buy_th}',
                    'shares': shares
                })

        current_total = cash + shares * price
        portfolio_values.append(current_total)
        daily_returns.append((current_total - prev_value) / prev_value if prev_value > 0 else 0)
        prev_value = current_total

    # 年末清仓
    if shares > 0:
        final_price = price_data.iloc[-1]['Close']
        final_sentiment = sent_data.iloc[-1]['smoothed_index'] if len(sent_data) > 0 else None
        final_ma = price_data.iloc[-1]['MA']
        sell_price = final_price * (1 - SLIPPAGE_RATE) * (1 - COMMISSION_RATE)
        cash += shares * sell_price
        trades.append({
            'type': 'SELL',
            'date': price_data.index[-1],
            'price': final_price,
            'net_price': sell_price,
            'sentiment': final_sentiment,
            'ma': final_ma,
            'reason': 'year_end',
            'shares': shares,
            'entry_date': entry_date,
            'entry_price': entry_price,
            'entry_sentiment': entry_sentiment,
            'trade_return': (sell_price - entry_price) / entry_price
        })

    final_value = cash
    total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL

    # 最大回撤
    if len(portfolio_values) > 0:
        portfolio_series = pd.Series(portfolio_values)
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

    return {
        'year': year,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'trades': trades,
        'final_value': final_value
    }


def main():
    print("=" * 120)
    print("年度独立回测 - 每年重置资金")
    print("=" * 120)

    loader = DataLoader(db_config)

    all_results = {}

    for symbol in OPTIMAL_CONFIG.keys():
        stock_data = loader.load_ohlcv(symbol, start_date="2014-01-01")
        sentiment_data = load_sentiment_data(symbol)

        if stock_data is None or len(stock_data) == 0:
            continue

        config = OPTIMAL_CONFIG[symbol]
        sell_desc = f">{config['sell_th']}" if config['sell_mode'] == 'threshold' else f">{config['sell_th']}&<MA"

        print(f"\n{'='*120}")
        print(f"{symbol} - 买入:<{config['buy_th']}, 卖出:{sell_desc}")
        print("=" * 120)

        results = []
        for year in TEST_YEARS:
            result = run_yearly_backtest(stock_data, sentiment_data, symbol, year)
            if result:
                results.append(result)

        all_results[symbol] = results

        # 打印年度汇总
        print(f"\n{'年份':<6} {'收益率':>10} {'最大回撤':>10} {'夏普率':>8} {'交易次数':>8}")
        print("-" * 50)

        for r in results:
            trade_count = len([t for t in r['trades'] if t['type'] == 'BUY'])
            print(f"{r['year']:<6} {r['total_return']:>10.2%} {r['max_drawdown']:>10.2%} "
                  f"{r['sharpe']:>8.2f} {trade_count:>8}")

        # 计算汇总
        avg_return = np.mean([r['total_return'] for r in results])
        avg_dd = np.mean([r['max_drawdown'] for r in results])
        avg_sharpe = np.mean([r['sharpe'] for r in results])
        cum_return = np.prod([1 + r['total_return'] for r in results]) - 1

        print("-" * 50)
        print(f"{'平均':<6} {avg_return:>10.2%} {avg_dd:>10.2%} {avg_sharpe:>8.2f}")
        print(f"{'累计':<6} {cum_return:>10.2%}")

        # 打印详细交易记录
        print(f"\n--- {symbol} 详细交易记录 ---")
        for r in results:
            if len(r['trades']) > 0:
                print(f"\n{r['year']}年:")
                for t in r['trades']:
                    if t['type'] == 'BUY':
                        print(f"  买入: {t['date'].strftime('%Y-%m-%d')} | "
                              f"价格=${t['price']:.2f} | "
                              f"情绪={t['sentiment']:.1f} | "
                              f"MA={t['ma']:.2f} | "
                              f"条件: {t['reason']} | "
                              f"股数={t['shares']}")
                    else:
                        print(f"  卖出: {t['date'].strftime('%Y-%m-%d')} | "
                              f"价格=${t['price']:.2f} | "
                              f"情绪={t['sentiment']:.1f} | "
                              f"MA={t['ma']:.2f} | "
                              f"条件: {t['reason']} | "
                              f"收益={t['trade_return']:+.2%}")

    loader.close()

    # 汇总所有股票
    print("\n" + "=" * 120)
    print("七姐妹汇总对比")
    print("=" * 120)

    print(f"\n{'股票':<8} {'平均收益':>10} {'5年累计':>12} {'平均回撤':>10} {'平均夏普':>10}")
    print("-" * 60)

    summary = []
    for symbol, results in all_results.items():
        avg_return = np.mean([r['total_return'] for r in results])
        cum_return = np.prod([1 + r['total_return'] for r in results]) - 1
        avg_dd = np.mean([r['max_drawdown'] for r in results])
        avg_sharpe = np.mean([r['sharpe'] for r in results])

        summary.append({
            'symbol': symbol,
            'avg_return': avg_return,
            'cum_return': cum_return,
            'avg_dd': avg_dd,
            'avg_sharpe': avg_sharpe
        })

    for s in sorted(summary, key=lambda x: x['cum_return'], reverse=True):
        print(f"{s['symbol']:<8} {s['avg_return']:>10.2%} {s['cum_return']:>12.2%} "
              f"{s['avg_dd']:>10.2%} {s['avg_sharpe']:>10.2f}")

    print("-" * 60)
    avg_all = np.mean([s['avg_return'] for s in summary])
    cum_all = np.mean([s['cum_return'] for s in summary])
    dd_all = np.mean([s['avg_dd'] for s in summary])
    sharpe_all = np.mean([s['avg_sharpe'] for s in summary])
    print(f"{'平均':<8} {avg_all:>10.2%} {cum_all:>12.2%} {dd_all:>10.2%} {sharpe_all:>10.2f}")

    # 年度横向对比
    print("\n" + "=" * 120)
    print("年度收益率对比")
    print("=" * 120)

    header = f"{'股票':<8}"
    for year in TEST_YEARS:
        header += f" {year:>10}"
    header += f" {'累计':>12}"
    print(header)
    print("-" * 75)

    for symbol, results in all_results.items():
        row = f"{symbol:<8}"
        for r in results:
            row += f" {r['total_return']:>10.2%}"
        cum = np.prod([1 + r['total_return'] for r in results]) - 1
        row += f" {cum:>12.2%}"
        print(row)

    print("\n" + "=" * 120)
    print("完成!")
    print("=" * 120)


if __name__ == "__main__":
    main()
