"""
2025年纯情绪策略交易可视化
显示：价格、情绪指数、买卖信号、资金曲线
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import psycopg2

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

# 2025年最优参数 (从Walk-Forward验证得出)
BUY_THRESHOLD = 0
SELL_THRESHOLD = 15

# 回测参数 (去掉止盈止损)
BACKTEST_PARAMS = {
    "initial_capital": 100000,
    "commission_rate": 0.001,
    "slippage_rate": 0.001,
    "take_profit_pct": 999.0,  # 禁用止盈
    "stop_loss_pct": 999.0,    # 禁用止损
    "max_holding_days": 999    # 禁用超时
}

POSITION_PCT = 0.8


def load_sentiment_data(db_config, symbol, start_date, end_date):
    """加载情绪数据"""
    conn = psycopg2.connect(
        host=db_config["host"],
        port=db_config["port"],
        database=db_config["database"],
        user=db_config["user"],
        password=db_config["password"]
    )

    query = f"""
        SELECT date, smoothed_index
        FROM fear_greed_index
        WHERE symbol = '{symbol}'
        AND date >= '{start_date}' AND date <= '{end_date}'
        ORDER BY date
    """

    df = pd.read_sql(query, conn)
    conn.close()

    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.set_index('date')

    return df


def run_backtest_2025():
    """运行2025年回测并返回详细数据"""
    print("加载数据...")

    # 加载价格数据
    loader = DataLoader(db_config)
    stock_data = loader.load_ohlcv(SYMBOL, start_date="2025-01-01", end_date="2025-12-31")
    loader.close()

    # 加载情绪数据
    sentiment_data = load_sentiment_data(db_config, SYMBOL, "2025-01-01", "2025-12-31")

    # 对齐数据
    common_idx = stock_data.index.intersection(sentiment_data.index)
    stock_data = stock_data.loc[common_idx]
    sentiment_data = sentiment_data.loc[common_idx]

    print(f"  价格数据: {len(stock_data)} 条")
    print(f"  情绪数据: {len(sentiment_data)} 条")
    print(f"  阈值: buy<{BUY_THRESHOLD}, sell>{SELL_THRESHOLD}")

    # 生成信号
    signals = pd.DataFrame(index=stock_data.index)
    signals['smoothed_index'] = sentiment_data['smoothed_index']
    signals['buy_signal'] = (signals['smoothed_index'] < BUY_THRESHOLD).astype(int)
    signals['sell_signal'] = (signals['smoothed_index'] > SELL_THRESHOLD).astype(int)
    signals['prob_profit'] = 0.5  # 纯情绪策略不使用概率
    signals['position_size'] = 0

    print(f"\n信号统计:")
    print(f"  买入信号天数: {signals['buy_signal'].sum()} ({signals['buy_signal'].mean()*100:.1f}%)")
    print(f"  卖出信号天数: {signals['sell_signal'].sum()} ({signals['sell_signal'].mean()*100:.1f}%)")

    # 回测
    backtester = EnhancedBacktester(
        initial_capital=BACKTEST_PARAMS["initial_capital"],
        commission_rate=BACKTEST_PARAMS["commission_rate"],
        slippage_rate=BACKTEST_PARAMS["slippage_rate"],
        take_profit_pct=BACKTEST_PARAMS["take_profit_pct"],
        stop_loss_pct=BACKTEST_PARAMS["stop_loss_pct"],
        max_holding_days=BACKTEST_PARAMS["max_holding_days"],
        use_dynamic_position=True,
        position_pct=POSITION_PCT
    )

    portfolio, metrics, trades = backtester.run_backtest_with_sell_signal(signals, stock_data)

    print(f"\n回测结果:")
    print(f"  总收益率: {metrics.get('total_return', 0):.2%}")
    print(f"  夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  最大回撤: {metrics.get('max_drawdown', 0):.2%}")
    print(f"  交易次数: {metrics.get('total_trades', 0)}")
    print(f"  胜率: {metrics.get('trade_win_rate', 0):.2%}")

    return stock_data, sentiment_data, signals, portfolio, trades, metrics


def create_visualization(stock_data, sentiment_data, signals, portfolio, trades, metrics):
    """创建4面板可视化"""

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(f'TSLA 2025年纯情绪策略交易分析\n阈值: buy<{BUY_THRESHOLD}, sell>{SELL_THRESHOLD} | '
                 f'收益: {metrics.get("total_return", 0):.2%} | 夏普: {metrics.get("sharpe_ratio", 0):.2f}',
                 fontsize=14, fontweight='bold')

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    # ============ 面板1: 价格与交易信号 ============
    ax1 = axes[0]
    ax1.plot(stock_data.index, stock_data['Close'], 'b-', linewidth=1.5, alpha=0.8, label='TSLA Price')
    ax1.set_ylabel('Price ($)', fontsize=11)
    ax1.set_title('价格与交易信号', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 标记买入卖出点
    if len(trades_df) > 0:
        for _, trade in trades_df.iterrows():
            entry_date = pd.to_datetime(trade['entry_date'])
            entry_price = trade['entry_price']

            # 买入点 (绿色向上三角)
            ax1.scatter(entry_date, entry_price, color='green', marker='^',
                       s=120, zorder=5, edgecolors='darkgreen', linewidths=1.5)

            # 卖出点
            if pd.notna(trade.get('exit_date')):
                exit_date = pd.to_datetime(trade['exit_date'])
                exit_price = trade['exit_price']
                exit_reason = trade.get('exit_reason', '')

                # 根据退出原因设置颜色
                if exit_reason == 'take_profit':
                    color, edge = 'red', 'darkred'
                elif exit_reason == 'stop_loss':
                    color, edge = 'orange', 'darkorange'
                elif exit_reason == 'sell_signal':
                    color, edge = 'blue', 'darkblue'
                else:
                    color, edge = 'purple', 'darkviolet'

                ax1.scatter(exit_date, exit_price, color=color, marker='v',
                           s=120, zorder=5, edgecolors=edge, linewidths=1.5)

    # 图例
    legend_elements = [
        Line2D([0], [0], color='b', linewidth=2, label='TSLA Price'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='green',
               markersize=10, markeredgecolor='darkgreen', label='Buy', linestyle='None'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='red',
               markersize=10, markeredgecolor='darkred', label='Take Profit', linestyle='None'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='orange',
               markersize=10, markeredgecolor='darkorange', label='Stop Loss', linestyle='None'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='blue',
               markersize=10, markeredgecolor='darkblue', label='Sell Signal', linestyle='None'),
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=9)

    # ============ 面板2: 情绪指数与阈值 ============
    ax2 = axes[1]
    ax2.plot(sentiment_data.index, sentiment_data['smoothed_index'],
             'purple', linewidth=1.5, alpha=0.8, label='Smoothed Index')
    ax2.axhline(y=BUY_THRESHOLD, color='green', linestyle='--', linewidth=1.5,
                label=f'Buy Threshold ({BUY_THRESHOLD})')
    ax2.axhline(y=SELL_THRESHOLD, color='red', linestyle='--', linewidth=1.5,
                label=f'Sell Threshold ({SELL_THRESHOLD})')
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    # 填充买入区域 (恐惧区)
    ax2.fill_between(sentiment_data.index, sentiment_data['smoothed_index'], BUY_THRESHOLD,
                     where=(sentiment_data['smoothed_index'] < BUY_THRESHOLD),
                     color='green', alpha=0.2, label='Buy Zone (Fear)')

    # 填充卖出区域 (贪婪区)
    ax2.fill_between(sentiment_data.index, sentiment_data['smoothed_index'], SELL_THRESHOLD,
                     where=(sentiment_data['smoothed_index'] > SELL_THRESHOLD),
                     color='red', alpha=0.2, label='Sell Zone (Greed)')

    ax2.set_ylabel('Sentiment Index', fontsize=11)
    ax2.set_title('情绪指数 (Smoothed Index)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-50, 50)

    # ============ 面板3: 买卖信号 ============
    ax3 = axes[2]

    # 买入信号区域
    buy_signal_dates = signals[signals['buy_signal'] == 1].index
    for date in buy_signal_dates:
        ax3.axvspan(date, date + pd.Timedelta(days=1), color='green', alpha=0.3)

    # 卖出信号区域
    sell_signal_dates = signals[signals['sell_signal'] == 1].index
    for date in sell_signal_dates:
        ax3.axvspan(date, date + pd.Timedelta(days=1), color='red', alpha=0.3)

    ax3.set_ylabel('Signal', fontsize=11)
    ax3.set_title(f'交易信号分布 (买入: {len(buy_signal_dates)}天, 卖出: {len(sell_signal_dates)}天)',
                  fontsize=12, fontweight='bold')
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['No Signal', 'Signal'])
    ax3.grid(True, alpha=0.3)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements3 = [
        Patch(facecolor='green', alpha=0.3, label=f'Buy Signal ({len(buy_signal_dates)} days)'),
        Patch(facecolor='red', alpha=0.3, label=f'Sell Signal ({len(sell_signal_dates)} days)')
    ]
    ax3.legend(handles=legend_elements3, loc='upper left', fontsize=9)

    # ============ 面板4: 资金曲线 ============
    ax4 = axes[3]
    ax4.plot(portfolio.index, portfolio['total_value'], 'b-', linewidth=2, label='Portfolio Value')
    ax4.axhline(y=BACKTEST_PARAMS["initial_capital"], color='gray', linestyle='--',
                linewidth=1, label='Initial Capital')

    # 标记最高点和最低点
    max_value = portfolio['total_value'].max()
    min_value = portfolio['total_value'].min()
    max_date = portfolio['total_value'].idxmax()
    min_date = portfolio['total_value'].idxmin()

    ax4.scatter(max_date, max_value, color='green', marker='*', s=200, zorder=5,
                label=f'Max: ${max_value:,.0f}')
    ax4.scatter(min_date, min_value, color='red', marker='*', s=200, zorder=5,
                label=f'Min: ${min_value:,.0f}')

    ax4.set_ylabel('Portfolio Value ($)', fontsize=11)
    ax4.set_xlabel('Date', fontsize=11)
    ax4.set_title(f'资金曲线 (最终: ${portfolio["total_value"].iloc[-1]:,.0f})',
                  fontsize=12, fontweight='bold')
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3)

    # 格式化x轴日期
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax4.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)

    plt.tight_layout()

    # 保存图片
    output_path = 'sentiment_strategy_2025.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: {output_path}")

    plt.close()

    return output_path


def main():
    print("=" * 60)
    print("TSLA 2025年纯情绪策略交易可视化")
    print("=" * 60)

    # 运行回测
    stock_data, sentiment_data, signals, portfolio, trades, metrics = run_backtest_2025()

    # 生成可视化
    print("\n生成可视化...")
    output_path = create_visualization(stock_data, sentiment_data, signals, portfolio, trades, metrics)

    # 打印交易明细
    if trades:
        print("\n" + "=" * 60)
        print("交易明细")
        print("=" * 60)
        trades_df = pd.DataFrame(trades)
        print(f"\n{'入场日期':<12} {'入场价':>10} {'出场日期':<12} {'出场价':>10} {'收益率':>10} {'原因':<15}")
        print("-" * 75)
        for _, t in trades_df.iterrows():
            entry = str(t['entry_date'])[:10]
            exit_d = str(t.get('exit_date', ''))[:10] if pd.notna(t.get('exit_date')) else 'Open'
            exit_p = f"${t['exit_price']:.2f}" if pd.notna(t.get('exit_price')) else '-'
            ret = f"{t.get('return_pct', 0):.2%}" if pd.notna(t.get('return_pct')) else '-'
            reason = t.get('exit_reason', '-')
            print(f"{entry:<12} ${t['entry_price']:>9.2f} {exit_d:<12} {exit_p:>10} {ret:>10} {reason:<15}")

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
