#!/usr/bin/env python3
"""
两层仓位管理回测结果可视化
生成5只股票的交易过程图
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from data_loader import DataLoader

# 数据库配置
db_config = {
    'host': 'localhost',
    'database': 'crypto_fear_greed_2',
    'user': 'sc2025',
    'password': '',
    'port': 5432
}

def visualize_backtest(trades_file, test_start='2016-01-01', test_end='2020-12-31'):
    """
    可视化回测结果

    Parameters:
    -----------
    trades_file : str
        交易记录CSV文件路径
    test_start : str
        测试开始日期
    test_end : str
        测试结束日期
    """

    # 读取交易记录
    trades_df = pd.read_csv(trades_file)
    trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'], utc=True)
    trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'], utc=True)

    # 获取股票列表
    symbols = ['MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AAPL']

    # 加载价格数据
    print("加载价格数据...")
    loader = DataLoader(db_config)

    prices_data = {}
    for symbol in symbols:
        prices = loader.load_ohlcv(symbol, test_start, test_end)
        prices_data[symbol] = prices
        print(f"  {symbol}: {len(prices)} 个交易日")

    # 创建图表 (5行1列)
    fig, axes = plt.subplots(5, 1, figsize=(20, 24))
    fig.suptitle('两层仓位管理 + 动态复利回测结果 (2016-2020年)',
                 fontsize=18, fontweight='bold', y=0.995)

    # 退出原因的颜色和标记
    exit_colors = {
        'OR': 'red',
        'AND': 'blue',
        'EOD': 'gray'
    }

    exit_labels = {
        'OR': 'OR退出 (指数>阈值)',
        'AND': 'AND退出 (指数高+破MA50)',
        'EOD': 'EOD退出 (期末平仓)'
    }

    # 为每只股票绘制子图
    for idx, symbol in enumerate(symbols):
        ax = axes[idx]
        prices = prices_data[symbol]

        # 绘制价格曲线
        ax.plot(prices.index, prices['Close'],
               color='steelblue', linewidth=1.5, alpha=0.8, label=f'{symbol} 价格')

        # 获取该股票的交易记录
        symbol_trades = trades_df[trades_df['symbol'] == symbol]

        # 绘制买入点
        for _, trade in symbol_trades.iterrows():
            entry_date = trade['entry_date']
            entry_price = trade['entry_price']

            # 买入标记（绿色向上三角）
            ax.scatter(entry_date, entry_price,
                      color='green', marker='^', s=200,
                      zorder=5, edgecolors='darkgreen', linewidths=2,
                      label='买入' if _ == symbol_trades.index[0] else '')

        # 绘制卖出点
        exit_reasons_seen = set()
        for _, trade in symbol_trades.iterrows():
            exit_date = trade['exit_date']
            exit_price = trade['exit_price']
            exit_reason = trade['exit_reason']

            # 卖出标记（不同颜色向下三角）
            color = exit_colors.get(exit_reason, 'orange')
            label = exit_labels.get(exit_reason, exit_reason)

            # 只为每个退出原因添加一次图例
            show_label = exit_reason not in exit_reasons_seen
            exit_reasons_seen.add(exit_reason)

            ax.scatter(exit_date, exit_price,
                      color=color, marker='v', s=200,
                      zorder=5, edgecolors='dark'+color if color != 'gray' else 'black',
                      linewidths=2,
                      label=label if show_label else '')

            # 绘制持仓期间的连线
            ax.plot([trade['entry_date'], exit_date],
                   [trade['entry_price'], exit_price],
                   color='gray', linestyle='--', alpha=0.3, linewidth=1)

        # 设置标题和标签
        # 计算该股票的统计信息
        if len(symbol_trades) > 0:
            total_profit = symbol_trades['profit'].sum()
            win_rate = len(symbol_trades[symbol_trades['profit'] > 0]) / len(symbol_trades) * 100
            num_trades = len(symbol_trades)
            avg_profit_pct = symbol_trades['profit_pct'].mean()

            title = f"{symbol} - 交易{num_trades}次 | 胜率{win_rate:.1f}% | 总盈亏${total_profit:,.0f} | 平均单笔{avg_profit_pct:+.1f}%"
        else:
            title = f"{symbol} - 无交易"

        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_ylabel('价格 ($)', fontsize=12)
        ax.grid(True, alpha=0.3)

        # 设置x轴日期格式
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.YearLocator())

        # 添加图例
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

        # 旋转日期标签
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 最后一个子图显示x轴标签
    axes[-1].set_xlabel('日期', fontsize=12)

    # 调整布局
    plt.tight_layout()

    # 保存图表
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'backtest_visualization_2layer_{timestamp}.png')

    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ 可视化图表已保存: {output_file}")

    plt.close()

    return output_file

def main():
    # 使用最新的交易记录文件
    trades_file = '/Users/sc2025/Desktop/test/AAPL/sentiment_strategy/results/trades_2layer_20260120_183116.csv'

    print("="*80)
    print("两层仓位管理回测可视化")
    print("="*80 + "\n")

    visualize_backtest(trades_file, test_start='2016-01-01', test_end='2020-12-31')

    print("\n" + "="*80)
    print("✅ 可视化完成！")
    print("="*80)

if __name__ == '__main__':
    main()
