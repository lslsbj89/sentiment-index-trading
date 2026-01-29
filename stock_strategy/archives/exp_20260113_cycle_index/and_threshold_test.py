#!/usr/bin/env python3
"""
测试降低AND卖出阈值对回撤的影响
原参数: AND>20, OR>40
测试: AND>10, AND>15 等
"""

import sys
sys.path.insert(0, '/Users/sc2025/Desktop/Claude/Quantrade/src')

import pandas as pd
import numpy as np
import psycopg2
from data_loader import DataLoader
import warnings
warnings.filterwarnings('ignore')

db_config = {
    'host': 'localhost',
    'database': 'crypto_fear_greed_2',
    'user': 'sc2025',
    'password': '',
    'port': 5432
}

SYMBOLS = ['NVDA', 'TSLA', 'AAPL', 'META', 'GOOGL', 'MSFT', 'AMZN']
BUY_THRESHOLD = 5
INITIAL_CAPITAL = 100000
POSITION_PCT = 0.8


def load_fear_greed_index(symbol):
    conn = psycopg2.connect(**db_config)
    query = f"""
        SELECT date, smoothed_index as idx
        FROM fear_greed_index
        WHERE symbol = '{symbol}'
        ORDER BY date
    """
    df = pd.read_sql(query, conn, parse_dates=['date'])
    conn.close()
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df.set_index('date', inplace=True)
    return df


def run_backtest(prices, index_data, and_threshold, or_threshold):
    df = prices.copy()
    df['idx'] = index_data['idx']
    df['MA50'] = df['Close'].rolling(50).mean()
    df = df.dropna()
    df = df[df.index >= '2021-01-01']

    if len(df) < 100:
        return None

    cash = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    trades = []
    portfolio_values = []

    for i in range(len(df)):
        current_price = df['Close'].iloc[i]
        current_idx = df['idx'].iloc[i]
        current_ma50 = df['MA50'].iloc[i]

        if position == 0 and current_idx < BUY_THRESHOLD:
            shares = int(cash * POSITION_PCT / (current_price * 1.002))
            if shares > 0:
                cash -= shares * current_price * 1.002
                position = shares
                entry_price = current_price

        elif position > 0:
            sell = False
            if current_idx > or_threshold:
                sell = True
            elif current_idx > and_threshold and current_price < current_ma50:
                sell = True

            if sell:
                revenue = position * current_price * 0.998
                trades.append((current_price - entry_price) / entry_price * 100)
                cash += revenue
                position = 0

        portfolio_values.append(cash + position * current_price)

    if position > 0:
        final_price = df['Close'].iloc[-1]
        trades.append((final_price - entry_price) / entry_price * 100)
        portfolio_values[-1] = cash + position * final_price * 0.998

    portfolio_series = pd.Series(portfolio_values)
    total_return = (portfolio_values[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    rolling_max = portfolio_series.cummax()
    max_drawdown = ((portfolio_series - rolling_max) / rolling_max * 100).min()
    returns = portfolio_series.pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    win_rate = sum(1 for t in trades if t > 0) / len(trades) * 100 if trades else 0

    return {
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'num_trades': len(trades),
        'win_rate': win_rate
    }


def main():
    print("=" * 90)
    print("AND卖出阈值测试")
    print("=" * 90)
    print(f"买入: idx < {BUY_THRESHOLD}")
    print(f"测试: AND阈值 = 10, 15, 20 (配合 OR>40)")

    loader = DataLoader(db_config)
    all_data = {}
    for symbol in SYMBOLS:
        prices = loader.load_ohlcv(symbol, start_date='2020-01-01', end_date='2025-12-31')
        index_data = load_fear_greed_index(symbol)
        if len(prices) > 0 and len(index_data) > 0:
            all_data[symbol] = {'prices': prices, 'index': index_data}
    loader.close()

    # 测试不同AND阈值
    and_thresholds = [10, 12, 15, 18, 20]
    or_threshold = 40

    print("\n" + "=" * 90)
    print("各AND阈值对比 (OR=40固定):")
    print("=" * 90)
    print(f"{'AND阈值':<10} {'收益率':<12} {'夏普率':<10} {'回撤':<12} {'胜率':<10} {'交易数':<10}")
    print("-" * 70)

    for and_th in and_thresholds:
        results = []
        for symbol, data in all_data.items():
            r = run_backtest(data['prices'], data['index'], and_th, or_threshold)
            if r:
                results.append(r)

        avg_return = np.mean([r['total_return'] for r in results])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
        avg_drawdown = np.mean([r['max_drawdown'] for r in results])
        avg_win_rate = np.mean([r['win_rate'] for r in results])
        avg_trades = np.mean([r['num_trades'] for r in results])

        print(f">{and_th:<8} {avg_return:>+8.1f}%  {avg_sharpe:>8.2f}  "
              f"{avg_drawdown:>8.1f}%  {avg_win_rate:>7.1f}%  {avg_trades:>8.1f}")

    # 详细对比 AND>10 vs AND>20
    print("\n" + "=" * 90)
    print("详细对比: AND>10 vs AND>20")
    print("=" * 90)
    print(f"{'股票':<8} {'---AND>20---':<25} {'---AND>10---':<25} {'回撤改善':<10}")
    print(f"{'':8} {'收益':>8} {'回撤':>8} {'夏普':>6} {'收益':>8} {'回撤':>8} {'夏普':>6} {'':<10}")
    print("-" * 85)

    for symbol, data in all_data.items():
        r20 = run_backtest(data['prices'], data['index'], 20, 40)
        r10 = run_backtest(data['prices'], data['index'], 10, 40)

        if r20 and r10:
            dd_improve = r20['max_drawdown'] - r10['max_drawdown']
            print(f"{symbol:<8} "
                  f"{r20['total_return']:>+7.1f}% {r20['max_drawdown']:>7.1f}% {r20['sharpe_ratio']:>6.2f} "
                  f"{r10['total_return']:>+7.1f}% {r10['max_drawdown']:>7.1f}% {r10['sharpe_ratio']:>6.2f} "
                  f"{dd_improve:>+7.1f}%")

    # 同时测试降低OR阈值
    print("\n" + "=" * 90)
    print("组合测试: AND + OR 同时调整")
    print("=" * 90)
    print(f"{'参数':<15} {'收益率':<12} {'夏普率':<10} {'回撤':<12} {'胜率':<10}")
    print("-" * 60)

    combos = [
        (20, 40, "AND>20,OR>40"),
        (15, 35, "AND>15,OR>35"),
        (10, 30, "AND>10,OR>30"),
        (10, 40, "AND>10,OR>40"),
        (15, 40, "AND>15,OR>40"),
    ]

    for and_th, or_th, label in combos:
        results = []
        for symbol, data in all_data.items():
            r = run_backtest(data['prices'], data['index'], and_th, or_th)
            if r:
                results.append(r)

        avg_return = np.mean([r['total_return'] for r in results])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
        avg_drawdown = np.mean([r['max_drawdown'] for r in results])
        avg_win_rate = np.mean([r['win_rate'] for r in results])

        print(f"{label:<15} {avg_return:>+8.1f}%  {avg_sharpe:>8.2f}  "
              f"{avg_drawdown:>8.1f}%  {avg_win_rate:>7.1f}%")


if __name__ == '__main__':
    main()
