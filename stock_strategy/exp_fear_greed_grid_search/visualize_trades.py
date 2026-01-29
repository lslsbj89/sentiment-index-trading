"""
交易过程可视化
显示价格、情绪指数、买卖信号、资金变化、回撤等
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import pandas as pd
import numpy as np
import psycopg2
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import datetime

from data_loader import DataLoader
from backtest_engine import EnhancedBacktester

# 配置
SYMBOL = "TSLA"
db_config = {
    "host": "localhost",
    "port": 5432,
    "database": "crypto_fear_greed_2",
    "user": "sc2025",
    "password": ""
}

# 最优参数
BUY_THRESHOLD = -15
AND_SELL_THRESHOLD = 25
OR_THRESHOLD = 45

TEST_START = "2021-01-01"
TEST_END = "2025-12-31"


def load_fear_greed_index(db_config, symbol):
    conn = psycopg2.connect(**db_config)
    query = """
        SELECT date, smoothed_index
        FROM fear_greed_index
        WHERE symbol = %s
        ORDER BY date
    """
    df = pd.read_sql(query, conn, params=(symbol,))
    conn.close()
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.set_index('date')
    df['smoothed_index'] = df['smoothed_index'].astype(float)
    return df


def load_price_with_ma(db_config, symbol):
    loader = DataLoader(db_config)
    ohlcv = loader.load_ohlcv(symbol, start_date="2015-01-01")
    loader.close()
    ohlcv['MA50'] = ohlcv['Close'].rolling(window=50).mean()
    return ohlcv


def main():
    print("生成可视化图表...")

    # 加载数据
    sentiment_data = load_fear_greed_index(db_config, SYMBOL)
    price_data = load_price_with_ma(db_config, SYMBOL)

    # 筛选时间范围
    start_ts = pd.Timestamp(TEST_START, tz='UTC')
    end_ts = pd.Timestamp(TEST_END, tz='UTC')
    mask = (price_data.index >= start_ts) & (price_data.index <= end_ts)
    test_price = price_data[mask].copy()
    test_sentiment = sentiment_data.reindex(test_price.index)

    # 构建信号
    signals = pd.DataFrame(index=test_price.index)
    signals['smoothed_index'] = test_sentiment['smoothed_index']
    signals['Close'] = test_price['Close']
    signals['MA50'] = test_price['MA50']

    signals['buy_signal'] = (signals['smoothed_index'] < BUY_THRESHOLD).astype(int)
    and_condition = (signals['smoothed_index'] > AND_SELL_THRESHOLD) & (signals['Close'] < signals['MA50'])
    or_condition = signals['smoothed_index'] > OR_THRESHOLD
    signals['sell_signal'] = (and_condition | or_condition).astype(int)
    signals['prob_profit'] = 0.5
    signals['position_size'] = 0

    # 回测
    backtester = EnhancedBacktester(
        initial_capital=100000,
        commission_rate=0.001,
        slippage_rate=0.001,
        take_profit_pct=999.0,
        stop_loss_pct=999.0,
        max_holding_days=999,
        use_dynamic_position=True,
        position_pct=0.8
    )

    portfolio, metrics, trades = backtester.run_backtest_with_sell_signal(signals, test_price)

    # 计算回撤
    cumulative = portfolio['total_value']
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100

    # 计算滚动夏普 (60日)
    returns = portfolio['total_value'].pct_change()
    rolling_sharpe = (returns.rolling(60).mean() / returns.rolling(60).std()) * np.sqrt(252)

    # ========================================
    # 创建可视化
    # ========================================
    fig = plt.figure(figsize=(20, 16))

    # 使用 GridSpec 创建布局
    gs = fig.add_gridspec(5, 4, height_ratios=[3, 2, 2, 2, 1.5],
                          hspace=0.3, wspace=0.3,
                          left=0.06, right=0.94, top=0.93, bottom=0.05)

    # ========================================
    # 子图1: 价格 + 买卖信号 (占4列)
    # ========================================
    ax1 = fig.add_subplot(gs[0, :])

    # 绘制价格
    ax1.plot(test_price.index, test_price['Close'], 'b-', linewidth=1.5, label='TSLA Price', alpha=0.8)
    ax1.plot(test_price.index, test_price['MA50'], 'orange', linewidth=1, label='MA50', alpha=0.7)

    # 标记持仓区间
    for trade in trades:
        entry = trade['entry_date']
        exit_dt = trade['exit_date']
        is_profit = trade['profit'] > 0
        color = 'lightgreen' if is_profit else 'lightcoral'
        ax1.axvspan(entry, exit_dt, alpha=0.2, color=color)

    # 标记买入点
    for trade in trades:
        entry = trade['entry_date']
        entry_price = trade['entry_price']
        ax1.scatter(entry, entry_price, marker='^', s=200, c='green',
                   edgecolors='darkgreen', linewidths=2, zorder=5, label='_nolegend_')
        ax1.annotate(f'BUY\n${entry_price:.0f}', xy=(entry, entry_price),
                    xytext=(10, 20), textcoords='offset points',
                    fontsize=9, fontweight='bold', color='green',
                    ha='left')

    # 标记卖出点
    for trade in trades:
        exit_dt = trade['exit_date']
        exit_price = trade['exit_price']
        exit_reason = trade['exit_reason']
        color = 'red' if exit_reason != 'open_position' else 'blue'
        marker_label = 'SELL' if exit_reason != 'open_position' else 'HOLD'
        ax1.scatter(exit_dt, exit_price, marker='v', s=200, c=color,
                   edgecolors='dark'+color if color=='red' else 'darkblue',
                   linewidths=2, zorder=5, label='_nolegend_')
        ax1.annotate(f'{marker_label}\n${exit_price:.0f}', xy=(exit_dt, exit_price),
                    xytext=(10, -30), textcoords='offset points',
                    fontsize=9, fontweight='bold', color=color,
                    ha='left')

    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title(f'{SYMBOL} Trading Strategy Visualization ({TEST_START} ~ {TEST_END})',
                  fontsize=16, fontweight='bold', pad=15)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    # ========================================
    # 子图2: 情绪指数 (占4列)
    # ========================================
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)

    # 绘制情绪指数
    ax2.fill_between(signals.index, 0, signals['smoothed_index'],
                     where=signals['smoothed_index'] > 0,
                     color='lightcoral', alpha=0.5, label='Greed (>0)')
    ax2.fill_between(signals.index, 0, signals['smoothed_index'],
                     where=signals['smoothed_index'] <= 0,
                     color='lightgreen', alpha=0.5, label='Fear (<0)')
    ax2.plot(signals.index, signals['smoothed_index'], 'k-', linewidth=1, alpha=0.7)

    # 阈值线
    ax2.axhline(y=BUY_THRESHOLD, color='green', linestyle='--', linewidth=2,
                label=f'Buy Threshold ({BUY_THRESHOLD})')
    ax2.axhline(y=AND_SELL_THRESHOLD, color='orange', linestyle='--', linewidth=2,
                label=f'AND Sell ({AND_SELL_THRESHOLD})')
    ax2.axhline(y=OR_THRESHOLD, color='red', linestyle='--', linewidth=2,
                label=f'OR Sell ({OR_THRESHOLD})')
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

    ax2.set_ylabel('Sentiment Index', fontsize=12)
    ax2.set_ylim(-60, 80)
    ax2.legend(loc='upper right', fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)

    # ========================================
    # 子图3: 组合价值 (占4列)
    # ========================================
    ax3 = fig.add_subplot(gs[2, :], sharex=ax1)

    ax3.fill_between(portfolio.index, 100000, portfolio['total_value'],
                     where=portfolio['total_value'] >= 100000,
                     color='lightgreen', alpha=0.5)
    ax3.fill_between(portfolio.index, 100000, portfolio['total_value'],
                     where=portfolio['total_value'] < 100000,
                     color='lightcoral', alpha=0.5)
    ax3.plot(portfolio.index, portfolio['total_value'], 'b-', linewidth=1.5)
    ax3.axhline(y=100000, color='gray', linestyle='--', linewidth=1, label='Initial Capital')

    # 标注最终价值
    final_value = portfolio['total_value'].iloc[-1]
    ax3.annotate(f'${final_value:,.0f}', xy=(portfolio.index[-1], final_value),
                xytext=(-60, 10), textcoords='offset points',
                fontsize=12, fontweight='bold', color='green')

    ax3.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

    # ========================================
    # 子图4: 回撤 (占2列)
    # ========================================
    ax4 = fig.add_subplot(gs[3, :2])

    ax4.fill_between(drawdown.index, 0, drawdown, color='red', alpha=0.4)
    ax4.plot(drawdown.index, drawdown, 'r-', linewidth=1)
    ax4.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

    # 标注最大回撤
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()
    ax4.scatter(max_dd_date, max_dd, s=100, c='darkred', zorder=5)
    ax4.annotate(f'Max DD: {max_dd:.1f}%', xy=(max_dd_date, max_dd),
                xytext=(10, -20), textcoords='offset points',
                fontsize=10, fontweight='bold', color='darkred')

    ax4.set_ylabel('Drawdown (%)', fontsize=12)
    ax4.set_xlabel('Date', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    # ========================================
    # 子图5: 滚动夏普比率 (占2列)
    # ========================================
    ax5 = fig.add_subplot(gs[3, 2:])

    ax5.plot(rolling_sharpe.index, rolling_sharpe, 'purple', linewidth=1.5)
    ax5.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax5.axhline(y=1, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Sharpe=1')
    ax5.axhline(y=2, color='darkgreen', linestyle='--', linewidth=1, alpha=0.5, label='Sharpe=2')

    ax5.fill_between(rolling_sharpe.index, 0, rolling_sharpe,
                     where=rolling_sharpe > 0, color='lightgreen', alpha=0.3)
    ax5.fill_between(rolling_sharpe.index, 0, rolling_sharpe,
                     where=rolling_sharpe <= 0, color='lightcoral', alpha=0.3)

    ax5.set_ylabel('Rolling Sharpe (60d)', fontsize=12)
    ax5.set_xlabel('Date', fontsize=12)
    ax5.legend(loc='upper right', fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    # ========================================
    # 子图6: 关键指标摘要 (占4列)
    # ========================================
    ax6 = fig.add_subplot(gs[4, :])
    ax6.axis('off')

    # 交易统计
    winning_trades = sum(1 for t in trades if t['profit'] > 0)
    losing_trades = sum(1 for t in trades if t['profit'] <= 0)
    total_profit = sum(t['profit'] for t in trades if t['profit'] > 0)
    total_loss = abs(sum(t['profit'] for t in trades if t['profit'] <= 0))
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    avg_holding = np.mean([t['holding_days'] for t in trades])

    # 创建指标文本
    metrics_text = f"""
    ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║  STRATEGY PARAMETERS                         PERFORMANCE METRICS                         TRADE STATISTICS            ║
    ║  ─────────────────────                       ───────────────────                         ────────────────            ║
    ║  Buy Threshold:    < {BUY_THRESHOLD:>4}                      Total Return:   {metrics['total_return']*100:>+8.2f}%                      Total Trades:    {len(trades):>4}            ║
    ║  AND Sell:         > {AND_SELL_THRESHOLD:>4} & < MA50              Annual Return:  {metrics['annualized_return']*100:>+8.2f}%                      Winning Trades:  {winning_trades:>4}            ║
    ║  OR Sell:          > {OR_THRESHOLD:>4}                      Sharpe Ratio:   {metrics['sharpe_ratio']:>8.2f}                       Losing Trades:   {losing_trades:>4}            ║
    ║  Position Size:     80%                      Max Drawdown:   {metrics['max_drawdown']*100:>8.2f}%                      Win Rate:      {winning_trades/len(trades)*100:>5.1f}%          ║
    ║  Initial Capital:  $100,000                  Final Value:    ${portfolio['total_value'].iloc[-1]:>10,.0f}                   Profit Factor:  {profit_factor:>5.2f}           ║
    ║                                                                                          Avg Holding:   {avg_holding:>5.0f} days       ║
    ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
    """

    ax6.text(0.5, 0.5, metrics_text, transform=ax6.transAxes,
             fontsize=11, fontfamily='monospace',
             verticalalignment='center', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 保存图表
    output_file = os.path.join(os.path.dirname(__file__), f'trading_visualization_{SYMBOL}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"可视化图表已保存: {output_file}")
    return output_file


if __name__ == "__main__":
    main()
