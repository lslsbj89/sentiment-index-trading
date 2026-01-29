"""
实验: fear_greed_index 网格搜索
使用 fear_greed_index 表的 smoothed_index 进行策略参数优化
找到最优参数后自动生成可视化图表
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from data_loader import DataLoader
from backtest_engine import EnhancedBacktester

# ============================================================
# 配置
# ============================================================
SYMBOL = "BABA"  # 目标股票

db_config = {
    "host": "localhost",
    "port": 5432,
    "database": "crypto_fear_greed_2",
    "user": "sc2025",
    "password": ""
}

# 网格搜索参数
BUY_THRESHOLDS = [-30, -25, -20, -15, -10, -5,0, 5]
AND_SELL_THRESHOLDS = [5, 10, 15, 20, 25]
OR_THRESHOLDS = [25, 30, 35, 40, 45, 50, 55, 60, 65]

# 回测参数 (无止盈止损，完全依赖情绪信号)
BACKTEST_PARAMS = {
    "initial_capital": 100000,
    "commission_rate": 0.001,
    "slippage_rate": 0.001,
    "take_profit_pct": 999.0,
    "stop_loss_pct": 999.0,
    "max_holding_days": 999,
    "position_pct": 0.8
}

# 回测时间范围
TEST_START = "2021-01-01"
TEST_END = "2025-12-31"


def load_fear_greed_index(db_config, symbol):
    """加载 fear_greed_index 表的 smoothed_index 数据"""
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
    """加载价格数据并计算MA50"""
    loader = DataLoader(db_config)
    ohlcv = loader.load_ohlcv(symbol, start_date="2015-01-01")
    loader.close()
    ohlcv['MA50'] = ohlcv['Close'].rolling(window=50).mean()
    return ohlcv


def run_single_backtest(price_data, sentiment_data, buy_threshold, and_sell_threshold,
                        or_threshold, start_date, end_date, use_dynamic=True):
    """运行单次回测"""
    start_ts = pd.Timestamp(start_date, tz='UTC')
    end_ts = pd.Timestamp(end_date, tz='UTC')

    mask = (price_data.index >= start_ts) & (price_data.index <= end_ts)
    test_price = price_data[mask].copy()
    test_sentiment = sentiment_data.reindex(test_price.index)

    if len(test_price) == 0:
        return None, None, None

    # 构建信号
    signals = pd.DataFrame(index=test_price.index)
    signals['smoothed_index'] = test_sentiment['smoothed_index']
    signals['Close'] = test_price['Close']
    signals['MA50'] = test_price['MA50']

    # 买入信号: 情绪 < buy_threshold
    signals['buy_signal'] = (signals['smoothed_index'] < buy_threshold).astype(int)

    # 卖出信号: (情绪 > and_sell_threshold AND 价格 < MA50) OR (情绪 > or_threshold)
    and_condition = (signals['smoothed_index'] > and_sell_threshold) & (signals['Close'] < signals['MA50'])
    or_condition = signals['smoothed_index'] > or_threshold
    signals['sell_signal'] = (and_condition | or_condition).astype(int)

    signals['prob_profit'] = 0.5
    signals['position_size'] = 0

    # 回测
    backtester = EnhancedBacktester(
        initial_capital=BACKTEST_PARAMS["initial_capital"],
        commission_rate=BACKTEST_PARAMS["commission_rate"],
        slippage_rate=BACKTEST_PARAMS["slippage_rate"],
        take_profit_pct=BACKTEST_PARAMS["take_profit_pct"],
        stop_loss_pct=BACKTEST_PARAMS["stop_loss_pct"],
        max_holding_days=BACKTEST_PARAMS["max_holding_days"],
        use_dynamic_position=use_dynamic,
        position_pct=BACKTEST_PARAMS["position_pct"]
    )

    portfolio, metrics, trades = backtester.run_backtest_with_sell_signal(signals, test_price)
    return portfolio, metrics, trades, signals


def visualize_trading(symbol, price_data, sentiment_data, portfolio, metrics, trades, signals,
                      buy_threshold, and_sell_threshold, or_threshold, test_start, test_end, output_dir):
    """生成交易可视化图表"""
    print("\n" + "=" * 80)
    print("生成可视化图表...")
    print("=" * 80)

    # 筛选时间范围
    start_ts = pd.Timestamp(test_start, tz='UTC')
    end_ts = pd.Timestamp(test_end, tz='UTC')
    mask = (price_data.index >= start_ts) & (price_data.index <= end_ts)
    test_price = price_data[mask].copy()

    # 计算回撤
    cumulative = portfolio['total_value']
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100

    # 计算滚动夏普 (60日)
    returns = portfolio['total_value'].pct_change()
    rolling_sharpe = (returns.rolling(60).mean() / returns.rolling(60).std()) * np.sqrt(252)

    # 创建可视化
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(5, 4, height_ratios=[3, 2, 2, 2, 1.5],
                          hspace=0.3, wspace=0.3,
                          left=0.06, right=0.94, top=0.93, bottom=0.05)

    # 子图1: 价格 + 买卖信号
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(test_price.index, test_price['Close'], 'b-', linewidth=1.5, label=f'{symbol} Price', alpha=0.8)
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
                   edgecolors='darkgreen', linewidths=2, zorder=5)
        ax1.annotate(f'BUY\n${entry_price:.0f}', xy=(entry, entry_price),
                    xytext=(10, 20), textcoords='offset points',
                    fontsize=9, fontweight='bold', color='green', ha='left')

    # 标记卖出点
    for trade in trades:
        exit_dt = trade['exit_date']
        exit_price = trade['exit_price']
        exit_reason = trade['exit_reason']
        color = 'red' if exit_reason != 'open_position' else 'blue'
        marker_label = 'SELL' if exit_reason != 'open_position' else 'HOLD'
        ax1.scatter(exit_dt, exit_price, marker='v', s=200, c=color,
                   edgecolors='dark'+color if color=='red' else 'darkblue',
                   linewidths=2, zorder=5)
        ax1.annotate(f'{marker_label}\n${exit_price:.0f}', xy=(exit_dt, exit_price),
                    xytext=(10, -30), textcoords='offset points',
                    fontsize=9, fontweight='bold', color=color, ha='left')

    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title(f'{symbol} Trading Strategy Visualization ({test_start} ~ {test_end})',
                  fontsize=16, fontweight='bold', pad=15)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    # 子图2: 情绪指数
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
    ax2.fill_between(signals.index, 0, signals['smoothed_index'],
                     where=signals['smoothed_index'] > 0,
                     color='lightcoral', alpha=0.5, label='Greed (>0)')
    ax2.fill_between(signals.index, 0, signals['smoothed_index'],
                     where=signals['smoothed_index'] <= 0,
                     color='lightgreen', alpha=0.5, label='Fear (<0)')
    ax2.plot(signals.index, signals['smoothed_index'], 'k-', linewidth=1, alpha=0.7)

    ax2.axhline(y=buy_threshold, color='green', linestyle='--', linewidth=2,
                label=f'Buy ({buy_threshold})')
    ax2.axhline(y=and_sell_threshold, color='orange', linestyle='--', linewidth=2,
                label=f'AND Sell ({and_sell_threshold})')
    ax2.axhline(y=or_threshold, color='red', linestyle='--', linewidth=2,
                label=f'OR Sell ({or_threshold})')
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

    ax2.set_ylabel('Sentiment Index', fontsize=12)
    ax2.set_ylim(-60, 80)
    ax2.legend(loc='upper right', fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)

    # 子图3: 组合价值
    ax3 = fig.add_subplot(gs[2, :], sharex=ax1)
    ax3.fill_between(portfolio.index, 100000, portfolio['total_value'],
                     where=portfolio['total_value'] >= 100000,
                     color='lightgreen', alpha=0.5)
    ax3.fill_between(portfolio.index, 100000, portfolio['total_value'],
                     where=portfolio['total_value'] < 100000,
                     color='lightcoral', alpha=0.5)
    ax3.plot(portfolio.index, portfolio['total_value'], 'b-', linewidth=1.5)
    ax3.axhline(y=100000, color='gray', linestyle='--', linewidth=1, label='Initial Capital')

    final_value = portfolio['total_value'].iloc[-1]
    ax3.annotate(f'${final_value:,.0f}', xy=(portfolio.index[-1], final_value),
                xytext=(-60, 10), textcoords='offset points',
                fontsize=12, fontweight='bold', color='green')

    ax3.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

    # 子图4: 回撤
    ax4 = fig.add_subplot(gs[3, :2])
    ax4.fill_between(drawdown.index, 0, drawdown, color='red', alpha=0.4)
    ax4.plot(drawdown.index, drawdown, 'r-', linewidth=1)
    ax4.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

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

    # 子图5: 滚动夏普比率
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

    # 子图6: 关键指标摘要
    ax6 = fig.add_subplot(gs[4, :])
    ax6.axis('off')

    winning_trades = sum(1 for t in trades if t['profit'] > 0)
    losing_trades = sum(1 for t in trades if t['profit'] <= 0)
    total_profit = sum(t['profit'] for t in trades if t['profit'] > 0)
    total_loss = abs(sum(t['profit'] for t in trades if t['profit'] <= 0))
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    avg_holding = np.mean([t['holding_days'] for t in trades]) if trades else 0

    metrics_text = f"""
    ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║  STRATEGY PARAMETERS                         PERFORMANCE METRICS                         TRADE STATISTICS            ║
    ║  ─────────────────────                       ───────────────────                         ────────────────            ║
    ║  Buy Threshold:    < {buy_threshold:>4}                      Total Return:   {metrics['total_return']*100:>+8.2f}%                      Total Trades:    {len(trades):>4}            ║
    ║  AND Sell:         > {and_sell_threshold:>4} & < MA50              Annual Return:  {metrics['annualized_return']*100:>+8.2f}%                      Winning Trades:  {winning_trades:>4}            ║
    ║  OR Sell:          > {or_threshold:>4}                      Sharpe Ratio:   {metrics['sharpe_ratio']:>8.2f}                       Losing Trades:   {losing_trades:>4}            ║
    ║  Position Size:     80%                      Max Drawdown:   {metrics['max_drawdown']*100:>8.2f}%                      Win Rate:      {winning_trades/len(trades)*100 if trades else 0:>5.1f}%          ║
    ║  Initial Capital:  $100,000                  Final Value:    ${portfolio['total_value'].iloc[-1]:>10,.0f}                   Profit Factor:  {profit_factor:>5.2f}           ║
    ║                                                                                          Avg Holding:   {avg_holding:>5.0f} days       ║
    ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
    """

    ax6.text(0.5, 0.5, metrics_text, transform=ax6.transAxes,
             fontsize=11, fontfamily='monospace',
             verticalalignment='center', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 保存图表
    output_file = os.path.join(output_dir, f'trading_visualization_{symbol}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"可视化图表已保存: {output_file}")
    return output_file


def main():
    print("=" * 80)
    print(f"实验: {SYMBOL} fear_greed_index 网格搜索")
    print("=" * 80)
    print(f"\n数据表: fear_greed_index (smoothed_index)")
    print(f"回测期间: {TEST_START} ~ {TEST_END}")
    print(f"\n搜索参数:")
    print(f"  买入阈值: {BUY_THRESHOLDS}")
    print(f"  AND卖出阈值: {AND_SELL_THRESHOLDS}")
    print(f"  OR兜底阈值: {OR_THRESHOLDS}")
    total_combinations = len(BUY_THRESHOLDS) * len(AND_SELL_THRESHOLDS) * len(OR_THRESHOLDS)
    print(f"  总组合数: {total_combinations}")

    # 加载数据
    print("\n加载数据...")
    sentiment_data = load_fear_greed_index(db_config, SYMBOL)
    price_data = load_price_with_ma(db_config, SYMBOL)
    print(f"情绪数据: {len(sentiment_data)} 行 ({sentiment_data.index.min().date()} ~ {sentiment_data.index.max().date()})")
    print(f"价格数据: {len(price_data)} 行")

    # 情绪指数统计
    print(f"\n情绪指数统计 (smoothed_index):")
    print(f"  均值: {sentiment_data['smoothed_index'].mean():.2f}")
    print(f"  标准差: {sentiment_data['smoothed_index'].std():.2f}")
    print(f"  最小值: {sentiment_data['smoothed_index'].min():.2f}")
    print(f"  最大值: {sentiment_data['smoothed_index'].max():.2f}")
    print(f"  25%分位: {sentiment_data['smoothed_index'].quantile(0.25):.2f}")
    print(f"  75%分位: {sentiment_data['smoothed_index'].quantile(0.75):.2f}")

    # 网格搜索
    print("\n" + "=" * 80)
    print("开始网格搜索...")
    print("=" * 80)

    results = []
    count = 0

    for buy_th in BUY_THRESHOLDS:
        for and_sell_th in AND_SELL_THRESHOLDS:
            for or_th in OR_THRESHOLDS:
                # 跳过无效组合：or_threshold 应该 >= and_sell_threshold
                if or_th < and_sell_th:
                    continue

                count += 1
                portfolio, metrics, trades, _ = run_single_backtest(
                    price_data, sentiment_data, buy_th, and_sell_th, or_th,
                    TEST_START, TEST_END
                )

                if metrics:
                    result = {
                        'buy_threshold': buy_th,
                        'and_sell_threshold': and_sell_th,
                        'or_threshold': or_th,
                        'total_return': metrics.get('total_return', 0) * 100,
                        'annualized_return': metrics.get('annualized_return', 0) * 100,
                        'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                        'max_drawdown': metrics.get('max_drawdown', 0) * 100,
                        'total_trades': metrics.get('total_trades', 0),
                        'win_rate': metrics.get('trade_win_rate', 0) * 100
                    }
                    results.append(result)

                    if count % 20 == 0 or count <= 5:
                        print(f"[{count:3d}] buy<{buy_th:3}, sell>{and_sell_th:2} AND <MA | OR>{or_th:2} | "
                              f"收益: {result['total_return']:8.2f}% | "
                              f"夏普: {result['sharpe_ratio']:5.2f} | "
                              f"回撤: {result['max_drawdown']:6.2f}% | "
                              f"交易: {result['total_trades']:3d}")

    # 转为DataFrame并排序
    results_df = pd.DataFrame(results)

    # 按夏普比率排序
    results_df = results_df.sort_values('sharpe_ratio', ascending=False)

    # 输出结果
    print("\n" + "=" * 80)
    print("网格搜索完成！")
    print("=" * 80)
    print(f"有效组合数: {len(results_df)}")

    # 最优参数 (按夏普)
    print("\n" + "-" * 80)
    print("TOP 10 最优参数组合 (按夏普比率)")
    print("-" * 80)

    top10 = results_df.head(10)
    for idx, (i, row) in enumerate(top10.iterrows(), 1):
        print(f"\n#{idx}:")
        print(f"  参数: buy < {row['buy_threshold']}, sell > {row['and_sell_threshold']} AND < MA50, OR > {row['or_threshold']}")
        print(f"  收益: {row['total_return']:.2f}% (年化: {row['annualized_return']:.2f}%)")
        print(f"  夏普: {row['sharpe_ratio']:.2f}")
        print(f"  回撤: {row['max_drawdown']:.2f}%")
        print(f"  交易: {int(row['total_trades'])}笔 (胜率: {row['win_rate']:.1f}%)")

    # 按收益排序的TOP10
    print("\n" + "-" * 80)
    print("TOP 10 最优参数组合 (按总收益)")
    print("-" * 80)

    top10_return = results_df.sort_values('total_return', ascending=False).head(10)
    for idx, (i, row) in enumerate(top10_return.iterrows(), 1):
        print(f"\n#{idx}:")
        print(f"  参数: buy < {row['buy_threshold']}, sell > {row['and_sell_threshold']} AND < MA50, OR > {row['or_threshold']}")
        print(f"  收益: {row['total_return']:.2f}% (年化: {row['annualized_return']:.2f}%)")
        print(f"  夏普: {row['sharpe_ratio']:.2f}")
        print(f"  回撤: {row['max_drawdown']:.2f}%")
        print(f"  交易: {int(row['total_trades'])}笔 (胜率: {row['win_rate']:.1f}%)")

    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(os.path.dirname(__file__), f"grid_search_{SYMBOL}_{timestamp}.csv")
    results_df.to_csv(output_file, index=False)
    print(f"\n结果已保存: {output_file}")

    # 保存最优参数
    best = results_df.iloc[0]
    summary_file = os.path.join(os.path.dirname(__file__), f"best_params_{SYMBOL}.txt")
    with open(summary_file, 'w') as f:
        f.write(f"实验: {SYMBOL} fear_greed_index 网格搜索\n")
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"回测期间: {TEST_START} ~ {TEST_END}\n\n")
        f.write(f"最优参数 (按夏普比率):\n")
        f.write(f"  buy_threshold: {best['buy_threshold']}\n")
        f.write(f"  and_sell_threshold: {best['and_sell_threshold']}\n")
        f.write(f"  or_threshold: {best['or_threshold']}\n\n")
        f.write(f"性能指标:\n")
        f.write(f"  总收益: {best['total_return']:.2f}%\n")
        f.write(f"  年化收益: {best['annualized_return']:.2f}%\n")
        f.write(f"  夏普比率: {best['sharpe_ratio']:.2f}\n")
        f.write(f"  最大回撤: {best['max_drawdown']:.2f}%\n")
        f.write(f"  交易次数: {int(best['total_trades'])}\n")
        f.write(f"  胜率: {best['win_rate']:.1f}%\n")
    print(f"最优参数已保存: {summary_file}")

    # 用最优参数运行并可视化
    best_buy = int(best['buy_threshold'])
    best_and_sell = int(best['and_sell_threshold'])
    best_or = int(best['or_threshold'])

    print(f"\n使用最优参数运行回测并生成可视化...")
    print(f"  buy < {best_buy}, sell > {best_and_sell} AND < MA50, OR > {best_or}")

    portfolio, metrics, trades, signals = run_single_backtest(
        price_data, sentiment_data, best_buy, best_and_sell, best_or,
        TEST_START, TEST_END
    )

    if portfolio is not None and len(trades) > 0:
        output_dir = os.path.dirname(__file__)
        visualize_trading(
            SYMBOL, price_data, sentiment_data, portfolio, metrics, trades, signals,
            best_buy, best_and_sell, best_or, TEST_START, TEST_END, output_dir
        )

        # 打印交易详情
        print("\n" + "-" * 80)
        print(f"交易详情 (共 {len(trades)} 笔)")
        print("-" * 80)
        for i, trade in enumerate(trades, 1):
            entry_date = trade['entry_date'].strftime('%Y-%m-%d')
            exit_date = trade['exit_date'].strftime('%Y-%m-%d')
            profit = trade['profit']
            profit_pct = trade['profit_pct'] * 100
            holding_days = trade['holding_days']
            exit_reason = trade['exit_reason']
            print(f"  #{i}: {entry_date} → {exit_date} | "
                  f"${trade['entry_price']:.2f} → ${trade['exit_price']:.2f} | "
                  f"{holding_days:3d}天 | {'+' if profit > 0 else ''}{profit:,.0f} ({profit_pct:+.1f}%) | {exit_reason}")

    return results_df


if __name__ == "__main__":
    results = main()
