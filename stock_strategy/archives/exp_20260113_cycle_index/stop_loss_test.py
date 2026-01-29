#!/usr/bin/env python3
"""
测试不同止损水平对策略表现的影响
基于 fear_greed_index 统一参数
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

# 数据库配置
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

# 回测参数
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
        AND date >= '2021-01-01' AND date <= '2025-12-31'
        ORDER BY date
    """
    df = pd.read_sql(query, conn, parse_dates=['date'])
    conn.close()
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df.set_index('date', inplace=True)
    return df


def run_backtest_with_stop_loss(prices, index_data, stop_loss_pct=None):
    """运行带止损的回测"""
    df = prices.copy()
    df['idx'] = index_data['fear_greed_index']
    df['MA50'] = df['Close'].rolling(50).mean()
    df = df.dropna()

    if len(df) < 100:
        return None

    cash = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    trades = []
    portfolio_values = [INITIAL_CAPITAL]

    for i in range(1, len(df)):
        current_price = df['Close'].iloc[i]
        current_idx = df['idx'].iloc[i]
        current_ma50 = df['MA50'].iloc[i]
        current_date = df.index[i]

        # 买入条件
        if position == 0 and current_idx < BUY_THRESHOLD:
            available = cash * POSITION_PCT
            shares = int(available / (current_price * (1 + COMMISSION + SLIPPAGE)))
            if shares > 0:
                cost = shares * current_price * (1 + COMMISSION + SLIPPAGE)
                cash -= cost
                position = shares
                entry_price = current_price
                entry_date = current_date

        # 卖出条件
        elif position > 0:
            sell_signal = False
            exit_reason = ''
            current_return = (current_price - entry_price) / entry_price

            # 止损检查 (优先级最高)
            if stop_loss_pct is not None and current_return <= -stop_loss_pct:
                sell_signal = True
                exit_reason = f'stop_loss_{int(stop_loss_pct*100)}%'
            # OR条件
            elif current_idx > OR_THRESHOLD:
                sell_signal = True
                exit_reason = f'idx>{OR_THRESHOLD}'
            # AND条件
            elif current_idx > AND_THRESHOLD and current_price < current_ma50:
                sell_signal = True
                exit_reason = f'idx>{AND_THRESHOLD} & <MA50'

            if sell_signal:
                revenue = position * current_price * (1 - COMMISSION - SLIPPAGE)
                profit = revenue - (position * entry_price * (1 + COMMISSION + SLIPPAGE))
                trades.append({
                    'profit': profit,
                    'profit_pct': current_return * 100,
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
        current_return = (final_price - entry_price) / entry_price
        trades.append({
            'profit': revenue - (position * entry_price * (1 + COMMISSION + SLIPPAGE)),
            'profit_pct': current_return * 100,
            'exit_reason': 'end_of_period'
        })
        cash += revenue
        portfolio_values[-1] = cash

    # 计算指标
    portfolio_series = pd.Series(portfolio_values)
    final_value = portfolio_values[-1]
    total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    rolling_max = portfolio_series.cummax()
    drawdown = (portfolio_series - rolling_max) / rolling_max * 100
    max_drawdown = drawdown.min()

    returns = portfolio_series.pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    wins = sum(1 for t in trades if t['profit'] > 0)
    win_rate = wins / len(trades) * 100 if trades else 0

    # 统计止损次数
    stop_loss_count = sum(1 for t in trades if 'stop_loss' in t['exit_reason'])

    return {
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'stop_loss_count': stop_loss_count
    }


def main():
    print("=" * 90)
    print("止损水平对策略表现的影响测试")
    print("=" * 90)
    print(f"\n基础策略: buy<{BUY_THRESHOLD}, AND>{AND_THRESHOLD}, OR>{OR_THRESHOLD}")

    loader = DataLoader(db_config)

    # 加载数据
    all_data = {}
    for symbol in SYMBOLS:
        prices = loader.load_ohlcv(symbol, start_date='2021-01-01', end_date='2025-12-31')
        index_data = load_fear_greed_index(symbol)
        if len(prices) > 0 and len(index_data) > 0:
            all_data[symbol] = {'prices': prices, 'index': index_data}
    loader.close()

    # 测试不同止损水平
    stop_loss_levels = [None, 0.05, 0.07, 0.10, 0.15, 0.20]

    print("\n" + "=" * 90)
    print("各止损水平对比:")
    print("=" * 90)
    print(f"{'止损':<10} {'收益率':<12} {'夏普率':<10} {'回撤':<12} {'胜率':<10} {'止损次数':<10}")
    print("-" * 70)

    results_by_level = []

    for sl in stop_loss_levels:
        sl_name = f"{int(sl*100)}%" if sl else "无"

        symbol_results = []
        total_sl_count = 0

        for symbol, data in all_data.items():
            result = run_backtest_with_stop_loss(
                data['prices'], data['index'], stop_loss_pct=sl
            )
            if result:
                symbol_results.append(result)
                total_sl_count += result['stop_loss_count']

        if symbol_results:
            avg_return = np.mean([r['total_return'] for r in symbol_results])
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in symbol_results])
            avg_drawdown = np.mean([r['max_drawdown'] for r in symbol_results])
            avg_win_rate = np.mean([r['win_rate'] for r in symbol_results])

            print(f"{sl_name:<10} {avg_return:>+8.1f}%  {avg_sharpe:>8.2f}  "
                  f"{avg_drawdown:>8.1f}%  {avg_win_rate:>7.1f}%  {total_sl_count:>8}")

            results_by_level.append({
                'stop_loss': sl_name,
                'avg_return': avg_return,
                'avg_sharpe': avg_sharpe,
                'avg_drawdown': avg_drawdown,
                'avg_win_rate': avg_win_rate,
                'stop_loss_count': total_sl_count
            })

    # 详细对比: 无止损 vs 10%止损
    print("\n" + "=" * 90)
    print("详细对比: 无止损 vs 10%止损")
    print("=" * 90)
    print(f"{'股票':<8} {'---无止损---':<25} {'---10%止损---':<25} {'回撤改善':<10}")
    print(f"{'':8} {'收益':>8} {'回撤':>8} {'夏普':>6} {'收益':>8} {'回撤':>8} {'夏普':>6} {'':<10}")
    print("-" * 80)

    for symbol, data in all_data.items():
        no_sl = run_backtest_with_stop_loss(data['prices'], data['index'], stop_loss_pct=None)
        sl_10 = run_backtest_with_stop_loss(data['prices'], data['index'], stop_loss_pct=0.10)

        if no_sl and sl_10:
            dd_improve = no_sl['max_drawdown'] - sl_10['max_drawdown']
            print(f"{symbol:<8} "
                  f"{no_sl['total_return']:>+7.1f}% {no_sl['max_drawdown']:>7.1f}% {no_sl['sharpe_ratio']:>6.2f} "
                  f"{sl_10['total_return']:>+7.1f}% {sl_10['max_drawdown']:>7.1f}% {sl_10['sharpe_ratio']:>6.2f} "
                  f"{dd_improve:>+7.1f}%")

    # 结论
    print("\n" + "=" * 90)
    print("结论:")
    print("=" * 90)

    # 找出最优止损水平
    best_sharpe = max(results_by_level, key=lambda x: x['avg_sharpe'])
    best_return = max(results_by_level, key=lambda x: x['avg_return'])
    best_drawdown = max(results_by_level, key=lambda x: x['avg_drawdown'])

    print(f"\n最高夏普率: {best_sharpe['stop_loss']} (夏普={best_sharpe['avg_sharpe']:.2f})")
    print(f"最高收益率: {best_return['stop_loss']} (收益={best_return['avg_return']:.1f}%)")
    print(f"最低回撤: {best_drawdown['stop_loss']} (回撤={best_drawdown['avg_drawdown']:.1f}%)")


if __name__ == '__main__':
    main()
