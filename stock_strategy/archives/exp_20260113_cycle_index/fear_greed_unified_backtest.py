#!/usr/bin/env python3
"""
fear_greed_index 统一参数回测与可视化
使用找到的最优统一参数：
  - 买入: idx < 5
  - AND卖出: idx > 20 AND price < MA50
  - OR卖出: idx > 40
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, '/Users/sc2025/Desktop/Claude/Quantrade/src')

import pandas as pd
import numpy as np
import psycopg2
import matplotlib.pyplot as plt
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

# MAG7股票
SYMBOLS = ['NVDA', 'TSLA', 'AAPL', 'META', 'GOOGL', 'MSFT', 'AMZN']

# 最优统一参数 (来自网格搜索)
BUY_THRESHOLD = 5      # 买入: idx < 5
AND_THRESHOLD = 20     # AND卖出: idx > 20 AND price < MA50
OR_THRESHOLD = 40      # OR卖出: idx > 40

# 回测参数
INITIAL_CAPITAL = 100000
POSITION_PCT = 0.8
COMMISSION = 0.001
SLIPPAGE = 0.001


def load_fear_greed_index(symbol):
    """加载 fear_greed_index 数据"""
    conn = psycopg2.connect(**db_config)
    query = f"""
        SELECT date, smoothed_index as fear_greed_index
        FROM fear_greed_index
        WHERE symbol = '{symbol}'
        AND date >= '2021-01-01'
        AND date <= '2025-12-31'
        ORDER BY date
    """
    df = pd.read_sql(query, conn, parse_dates=['date'])
    conn.close()
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df.set_index('date', inplace=True)
    return df


def run_backtest_with_details(prices, index_data):
    """运行回测并返回详细交易记录"""
    # 合并数据
    df = prices.copy()
    df['idx'] = index_data['fear_greed_index']
    df['MA50'] = df['Close'].rolling(50).mean()
    df = df.dropna()

    if len(df) < 100:
        return None, None, None, None

    # 初始化
    cash = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    trades = []
    portfolio_values = []
    drawdowns = []

    for i in range(len(df)):
        current_price = df['Close'].iloc[i]
        current_idx = df['idx'].iloc[i]
        current_ma50 = df['MA50'].iloc[i]
        current_date = df.index[i]

        # 买入条件
        if position == 0 and current_idx < BUY_THRESHOLD:
            # 动态仓位
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

            # OR条件：指数 > OR_THRESHOLD
            if current_idx > OR_THRESHOLD:
                sell_signal = True
                exit_reason = f'idx>{OR_THRESHOLD}'
            # AND条件：指数 > AND_THRESHOLD 且 价格 < MA50
            elif current_idx > AND_THRESHOLD and current_price < current_ma50:
                sell_signal = True
                exit_reason = f'idx>{AND_THRESHOLD} & <MA50'

            if sell_signal:
                revenue = position * current_price * (1 - COMMISSION - SLIPPAGE)
                profit = revenue - (position * entry_price * (1 + COMMISSION + SLIPPAGE))
                profit_pct = (current_price - entry_price) / entry_price * 100
                holding_days = (current_date - entry_date).days

                trades.append({
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'entry_idx': entry_idx,
                    'exit_date': current_date,
                    'exit_price': current_price,
                    'exit_idx': current_idx,
                    'shares': position,
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'holding_days': holding_days,
                    'exit_reason': exit_reason
                })

                cash += revenue
                position = 0

        # 记录组合价值
        portfolio_value = cash + position * current_price
        portfolio_values.append(portfolio_value)

    # 期末强制平仓
    if position > 0:
        final_price = df['Close'].iloc[-1]
        revenue = position * final_price * (1 - COMMISSION - SLIPPAGE)
        profit = revenue - (position * entry_price * (1 + COMMISSION + SLIPPAGE))
        profit_pct = (final_price - entry_price) / entry_price * 100
        holding_days = (df.index[-1] - entry_date).days

        trades.append({
            'entry_date': entry_date,
            'entry_price': entry_price,
            'entry_idx': entry_idx,
            'exit_date': df.index[-1],
            'exit_price': final_price,
            'exit_idx': df['idx'].iloc[-1],
            'shares': position,
            'profit': profit,
            'profit_pct': profit_pct,
            'holding_days': holding_days,
            'exit_reason': 'end_of_period'
        })

        cash += revenue
        portfolio_values[-1] = cash

    # 计算指标
    portfolio_series = pd.Series(portfolio_values, index=df.index)
    final_value = portfolio_values[-1]
    total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    # 最大回撤
    rolling_max = portfolio_series.cummax()
    drawdown = (portfolio_series - rolling_max) / rolling_max * 100
    max_drawdown = drawdown.min()

    # 夏普率
    returns = portfolio_series.pct_change().dropna()
    if len(returns) > 0 and returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
    else:
        sharpe = 0

    # 胜率
    if len(trades) > 0:
        wins = sum(1 for t in trades if t['profit'] > 0)
        win_rate = wins / len(trades) * 100
    else:
        win_rate = 0

    metrics = {
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'final_value': final_value
    }

    return df, portfolio_series, drawdown, trades, metrics


def plot_trades(symbol, df, portfolio_series, drawdown, trades, metrics):
    """绘制交易图表"""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12),
                            gridspec_kw={'height_ratios': [2, 1, 1]})

    # 图1: 价格与交易信号
    ax1 = axes[0]
    ax1.plot(df.index, df['Close'], 'b-', linewidth=1, alpha=0.7, label='Price')
    ax1.plot(df.index, df['MA50'], 'orange', linewidth=1, linestyle='--', alpha=0.7, label='MA50')

    # 标记买入卖出点
    for trade in trades:
        # 买入点
        ax1.scatter(trade['entry_date'], trade['entry_price'],
                   color='green', marker='^', s=150, zorder=5,
                   edgecolors='darkgreen', linewidths=1.5)

        # 卖出点
        if trade['exit_reason'] == 'end_of_period':
            color = 'gray'
        elif trade['profit'] > 0:
            color = 'red'
        else:
            color = 'orange'

        ax1.scatter(trade['exit_date'], trade['exit_price'],
                   color=color, marker='v', s=150, zorder=5,
                   edgecolors='dark' + color if color != 'gray' else 'black', linewidths=1.5)

    ax1.set_title(f'{symbol} - fear_greed_index 统一参数策略 (buy<{BUY_THRESHOLD}, AND>{AND_THRESHOLD}, OR>{OR_THRESHOLD})\n'
                  f'收益: {metrics["total_return"]:.1f}% | 回撤: {metrics["max_drawdown"]:.1f}% | '
                  f'夏普: {metrics["sharpe_ratio"]:.2f} | 胜率: {metrics["win_rate"]:.0f}%',
                  fontsize=12)
    ax1.set_ylabel('Price', fontsize=10)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 图2: 情绪指数
    ax2 = axes[1]
    ax2.fill_between(df.index, 0, df['idx'], where=(df['idx'] > 0),
                    color='lightcoral', alpha=0.5, label='Greedy')
    ax2.fill_between(df.index, 0, df['idx'], where=(df['idx'] <= 0),
                    color='lightgreen', alpha=0.5, label='Fearful')
    ax2.plot(df.index, df['idx'], 'k-', linewidth=0.8)

    # 标记阈值线
    ax2.axhline(y=BUY_THRESHOLD, color='green', linestyle='--', alpha=0.7, label=f'Buy < {BUY_THRESHOLD}')
    ax2.axhline(y=AND_THRESHOLD, color='orange', linestyle='--', alpha=0.7, label=f'AND > {AND_THRESHOLD}')
    ax2.axhline(y=OR_THRESHOLD, color='red', linestyle='--', alpha=0.7, label=f'OR > {OR_THRESHOLD}')

    ax2.set_ylabel('Fear Greed Index', fontsize=10)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 图3: 组合价值和回撤
    ax3 = axes[2]
    ax3_twin = ax3.twinx()

    # 组合价值
    ax3.plot(portfolio_series.index, portfolio_series / 1000, 'b-', linewidth=1.5, label='Portfolio ($K)')
    ax3.fill_between(portfolio_series.index, INITIAL_CAPITAL / 1000, portfolio_series / 1000,
                    alpha=0.3, color='blue')
    ax3.axhline(y=INITIAL_CAPITAL / 1000, color='gray', linestyle='--', alpha=0.5)

    # 回撤
    ax3_twin.fill_between(drawdown.index, 0, drawdown, color='red', alpha=0.3, label='Drawdown')

    ax3.set_xlabel('Date', fontsize=10)
    ax3.set_ylabel('Portfolio Value ($K)', fontsize=10, color='blue')
    ax3_twin.set_ylabel('Drawdown (%)', fontsize=10, color='red')
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{symbol}_fear_greed_unified.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {symbol}_fear_greed_unified.png")


def main():
    print("=" * 80)
    print("fear_greed_index 统一参数回测")
    print("=" * 80)
    print(f"\n统一参数:")
    print(f"  买入: idx < {BUY_THRESHOLD}")
    print(f"  AND卖出: idx > {AND_THRESHOLD} AND price < MA50")
    print(f"  OR卖出: idx > {OR_THRESHOLD}")

    # 加载数据
    loader = DataLoader(db_config)

    all_trades = []
    all_metrics = []

    print("\n" + "=" * 80)
    print("回测结果:")
    print("=" * 80)
    print(f"{'股票':<8} {'收益率':<12} {'夏普率':<10} {'回撤':<12} {'交易数':<8} {'胜率':<8}")
    print("-" * 60)

    for symbol in SYMBOLS:
        prices = loader.load_ohlcv(symbol, start_date='2021-01-01', end_date='2025-12-31')
        index_data = load_fear_greed_index(symbol)

        if len(prices) == 0 or len(index_data) == 0:
            print(f"  {symbol}: 数据不足，跳过")
            continue

        result = run_backtest_with_details(prices, index_data)
        if result[0] is None:
            continue

        df, portfolio_series, drawdown, trades, metrics = result

        print(f"{symbol:<8} {metrics['total_return']:>+8.1f}%  "
              f"{metrics['sharpe_ratio']:>8.2f}  "
              f"{metrics['max_drawdown']:>8.1f}%  "
              f"{metrics['num_trades']:>6}  "
              f"{metrics['win_rate']:>5.1f}%")

        # 绘制图表
        plot_trades(symbol, df, portfolio_series, drawdown, trades, metrics)

        # 记录交易
        for t in trades:
            t['symbol'] = symbol
            all_trades.append(t)

        metrics['symbol'] = symbol
        all_metrics.append(metrics)

    loader.close()

    # 保存交易记录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    trades_df = pd.DataFrame(all_trades)
    trades_file = f'fear_greed_unified_trades_{timestamp}.csv'
    trades_df.to_csv(trades_file, index=False)

    summary_df = pd.DataFrame(all_metrics)
    summary_file = f'fear_greed_unified_summary_{timestamp}.csv'
    summary_df.to_csv(summary_file, index=False)

    # 汇总统计
    print("\n" + "=" * 80)
    print("汇总统计:")
    print("=" * 80)
    print(f"  平均收益率: {summary_df['total_return'].mean():+.1f}%")
    print(f"  平均夏普率: {summary_df['sharpe_ratio'].mean():.2f}")
    print(f"  平均回撤: {summary_df['max_drawdown'].mean():.1f}%")
    print(f"  平均胜率: {summary_df['win_rate'].mean():.1f}%")
    print(f"  总交易数: {len(all_trades)}")

    print(f"\n文件已保存:")
    print(f"  交易记录: {trades_file}")
    print(f"  汇总统计: {summary_file}")


if __name__ == '__main__':
    main()
