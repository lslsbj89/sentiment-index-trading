#!/usr/bin/env python3
"""
测试 MA200 过滤器对策略表现的影响
买入条件增加: price > MA200 (只在上升趋势中买入)
"""

import sys
import os
sys.path.insert(0, '/Users/sc2025/Desktop/Claude/Quantrade/src')

import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader

db_config = {
    'host': 'localhost',
    'database': 'crypto_fear_greed_2',
    'user': 'sc2025',
    'password': '',
    'port': 5432
}

SYMBOLS = ['NVDA', 'TSLA', 'AAPL', 'META', 'GOOGL', 'MSFT', 'AMZN']

# 策略参数
BUY_THRESHOLD = 5
AND_THRESHOLD = 20
OR_THRESHOLD = 40

INITIAL_CAPITAL = 100000
POSITION_PCT = 0.8
COMMISSION = 0.001
SLIPPAGE = 0.001


def load_fear_greed_index(symbol):
    conn = psycopg2.connect(**db_config)
    query = f"""
        SELECT date, smoothed_index as fear_greed_index
        FROM fear_greed_index
        WHERE symbol = '{symbol}'
        AND date >= '2020-01-01' AND date <= '2025-12-31'
        ORDER BY date
    """
    df = pd.read_sql(query, conn, parse_dates=['date'])
    conn.close()
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df.set_index('date', inplace=True)
    return df


def run_backtest(prices, index_data, use_ma200_filter=False):
    """运行回测"""
    df = prices.copy()
    df['idx'] = index_data['fear_greed_index']
    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()
    df = df.dropna()

    # 只取2021年之后的数据
    df = df[df.index >= '2021-01-01']

    if len(df) < 100:
        return None

    cash = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    trades = []
    portfolio_values = []

    # 记录买入被MA200过滤的次数
    ma200_filtered = 0

    for i in range(len(df)):
        current_price = df['Close'].iloc[i]
        current_idx = df['idx'].iloc[i]
        current_ma50 = df['MA50'].iloc[i]
        current_ma200 = df['MA200'].iloc[i]
        current_date = df.index[i]

        # 买入条件
        if position == 0 and current_idx < BUY_THRESHOLD:
            # MA200 过滤
            if use_ma200_filter and current_price < current_ma200:
                ma200_filtered += 1
                portfolio_values.append(cash)
                continue

            available = cash * POSITION_PCT
            shares = int(available / (current_price * (1 + COMMISSION + SLIPPAGE)))
            if shares > 0:
                cost = shares * current_price * (1 + COMMISSION + SLIPPAGE)
                cash -= cost
                position = shares
                entry_price = current_price
                entry_date = current_date
                entry_idx = current_idx

        # 卖出条件
        elif position > 0:
            sell_signal = False
            exit_reason = ''

            if current_idx > OR_THRESHOLD:
                sell_signal = True
                exit_reason = f'idx>{OR_THRESHOLD}'
            elif current_idx > AND_THRESHOLD and current_price < current_ma50:
                sell_signal = True
                exit_reason = f'idx>{AND_THRESHOLD} & <MA50'

            if sell_signal:
                revenue = position * current_price * (1 - COMMISSION - SLIPPAGE)
                profit_pct = (current_price - entry_price) / entry_price * 100
                holding_days = (current_date - entry_date).days
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'profit_pct': profit_pct,
                    'holding_days': holding_days,
                    'exit_reason': exit_reason
                })
                cash += revenue
                position = 0

        portfolio_value = cash + position * current_price
        portfolio_values.append(portfolio_value)

    # 期末平仓
    if position > 0:
        final_price = df['Close'].iloc[-1]
        revenue = position * final_price * (1 - COMMISSION - SLIPPAGE)
        profit_pct = (final_price - entry_price) / entry_price * 100
        holding_days = (df.index[-1] - entry_date).days
        trades.append({
            'entry_date': entry_date,
            'exit_date': df.index[-1],
            'entry_price': entry_price,
            'exit_price': final_price,
            'profit_pct': profit_pct,
            'holding_days': holding_days,
            'exit_reason': 'end_of_period'
        })
        cash += revenue
        portfolio_values[-1] = cash

    # 计算指标
    portfolio_series = pd.Series(portfolio_values, index=df.index[:len(portfolio_values)])
    final_value = portfolio_values[-1] if portfolio_values else INITIAL_CAPITAL
    total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    rolling_max = portfolio_series.cummax()
    drawdown = (portfolio_series - rolling_max) / rolling_max * 100
    max_drawdown = drawdown.min()

    returns = portfolio_series.pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 and returns.std() > 0 else 0

    wins = sum(1 for t in trades if t['profit_pct'] > 0)
    win_rate = wins / len(trades) * 100 if trades else 0

    return {
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'ma200_filtered': ma200_filtered,
        'trades': trades
    }


def main():
    print("=" * 90)
    print("MA200 过滤器测试")
    print("=" * 90)
    print(f"\n基础策略: buy<{BUY_THRESHOLD}, AND>{AND_THRESHOLD}, OR>{OR_THRESHOLD}")
    print(f"MA200过滤: 只在 price > MA200 时买入")

    loader = DataLoader(db_config)

    all_data = {}
    for symbol in SYMBOLS:
        prices = loader.load_ohlcv(symbol, start_date='2020-01-01', end_date='2025-12-31')
        index_data = load_fear_greed_index(symbol)
        if len(prices) > 0 and len(index_data) > 0:
            all_data[symbol] = {'prices': prices, 'index': index_data}
    loader.close()

    print("\n" + "=" * 90)
    print("各股票对比: 无过滤 vs MA200过滤")
    print("=" * 90)
    print(f"{'股票':<8} {'---无过滤---':<30} {'---MA200过滤---':<30} {'改善':<15}")
    print(f"{'':8} {'收益':>8} {'回撤':>8} {'夏普':>6} {'交易':>5} "
          f"{'收益':>8} {'回撤':>8} {'夏普':>6} {'交易':>5} {'回撤':>8}")
    print("-" * 95)

    no_filter_results = []
    ma200_results = []

    for symbol, data in all_data.items():
        no_filter = run_backtest(data['prices'], data['index'], use_ma200_filter=False)
        with_ma200 = run_backtest(data['prices'], data['index'], use_ma200_filter=True)

        if no_filter and with_ma200:
            dd_improve = no_filter['max_drawdown'] - with_ma200['max_drawdown']

            print(f"{symbol:<8} "
                  f"{no_filter['total_return']:>+7.1f}% {no_filter['max_drawdown']:>7.1f}% "
                  f"{no_filter['sharpe_ratio']:>6.2f} {no_filter['num_trades']:>5} "
                  f"{with_ma200['total_return']:>+7.1f}% {with_ma200['max_drawdown']:>7.1f}% "
                  f"{with_ma200['sharpe_ratio']:>6.2f} {with_ma200['num_trades']:>5} "
                  f"{dd_improve:>+7.1f}%")

            no_filter_results.append(no_filter)
            ma200_results.append(with_ma200)

    # 汇总
    print("-" * 95)
    avg_no = {
        'return': np.mean([r['total_return'] for r in no_filter_results]),
        'drawdown': np.mean([r['max_drawdown'] for r in no_filter_results]),
        'sharpe': np.mean([r['sharpe_ratio'] for r in no_filter_results]),
        'trades': np.mean([r['num_trades'] for r in no_filter_results])
    }
    avg_ma = {
        'return': np.mean([r['total_return'] for r in ma200_results]),
        'drawdown': np.mean([r['max_drawdown'] for r in ma200_results]),
        'sharpe': np.mean([r['sharpe_ratio'] for r in ma200_results]),
        'trades': np.mean([r['num_trades'] for r in ma200_results])
    }

    print(f"{'平均':<8} "
          f"{avg_no['return']:>+7.1f}% {avg_no['drawdown']:>7.1f}% "
          f"{avg_no['sharpe']:>6.2f} {avg_no['trades']:>5.1f} "
          f"{avg_ma['return']:>+7.1f}% {avg_ma['drawdown']:>7.1f}% "
          f"{avg_ma['sharpe']:>6.2f} {avg_ma['trades']:>5.1f} "
          f"{avg_no['drawdown'] - avg_ma['drawdown']:>+7.1f}%")

    # 结论
    print("\n" + "=" * 90)
    print("结论:")
    print("=" * 90)

    ret_change = avg_ma['return'] - avg_no['return']
    dd_change = avg_ma['drawdown'] - avg_no['drawdown']
    sharpe_change = avg_ma['sharpe'] - avg_no['sharpe']

    print(f"\n收益率变化: {ret_change:+.1f}% ({avg_no['return']:.1f}% → {avg_ma['return']:.1f}%)")
    print(f"回撤变化: {dd_change:+.1f}% ({avg_no['drawdown']:.1f}% → {avg_ma['drawdown']:.1f}%)")
    print(f"夏普变化: {sharpe_change:+.2f} ({avg_no['sharpe']:.2f} → {avg_ma['sharpe']:.2f})")

    if dd_change > 0:
        print(f"\nMA200过滤有效降低回撤 {-dd_change:.1f}%")
    else:
        print(f"\nMA200过滤未能降低回撤")

    # 分析被过滤的买入信号
    print("\n" + "=" * 90)
    print("被MA200过滤的买入信号:")
    print("=" * 90)

    total_filtered = sum(r['ma200_filtered'] for r in ma200_results)
    print(f"总共被过滤: {total_filtered} 次")


if __name__ == '__main__':
    main()
