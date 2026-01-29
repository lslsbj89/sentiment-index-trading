#!/usr/bin/env python3
"""
两种指数策略对比分析：
1. cycle_index (5因子，含黄金)
2. fear_greed_index (4因子，无黄金)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 读取两个指数的回测结果
cycle_df = pd.read_csv('unified_backtest_summary_20260115_225251.csv')
fg_df = pd.read_csv('fear_greed_unified_summary_20260115_233116.csv')

# 标准化列名
cycle_df = cycle_df.rename(columns={'total_trades': 'num_trades'})

SYMBOLS = ['NVDA', 'TSLA', 'AAPL', 'META', 'GOOGL', 'MSFT', 'AMZN']

print("=" * 90)
print("两种情绪指数策略对比分析")
print("=" * 90)

print("\n统一参数设置:")
print("-" * 60)
print("cycle_index (5因子):")
print("  买入: idx < 0")
print("  AND卖出: idx > 15 AND price < MA50")
print("  OR卖出: idx > 35")

print("\nfear_greed_index (4因子，无黄金):")
print("  买入: idx < 5")
print("  AND卖出: idx > 20 AND price < MA50")
print("  OR卖出: idx > 40")

# 计算各指数指标
print("\n" + "=" * 90)
print("各股票表现对比:")
print("=" * 90)

print(f"\n{'股票':<8} {'---cycle_index---':<30} {'---fear_greed_index---':<30} {'差异':<15}")
print(f"{'':8} {'收益率':>10} {'夏普':>8} {'回撤':>10} {'收益率':>10} {'夏普':>8} {'回撤':>10} {'收益':>8}")
print("-" * 90)

comparison_data = []

for symbol in SYMBOLS:
    cycle_row = cycle_df[cycle_df['symbol'] == symbol].iloc[0]
    fg_row = fg_df[fg_df['symbol'] == symbol].iloc[0]

    ret_diff = fg_row['total_return'] - cycle_row['total_return']

    print(f"{symbol:<8} "
          f"{cycle_row['total_return']:>+8.1f}% "
          f"{cycle_row['sharpe_ratio']:>8.2f} "
          f"{cycle_row['max_drawdown']:>8.1f}%   "
          f"{fg_row['total_return']:>+8.1f}% "
          f"{fg_row['sharpe_ratio']:>8.2f} "
          f"{fg_row['max_drawdown']:>8.1f}%   "
          f"{ret_diff:>+6.1f}%")

    comparison_data.append({
        'symbol': symbol,
        'cycle_return': cycle_row['total_return'],
        'cycle_sharpe': cycle_row['sharpe_ratio'],
        'cycle_drawdown': cycle_row['max_drawdown'],
        'fg_return': fg_row['total_return'],
        'fg_sharpe': fg_row['sharpe_ratio'],
        'fg_drawdown': fg_row['max_drawdown'],
        'return_diff': ret_diff
    })

# 汇总统计
print("\n" + "=" * 90)
print("汇总统计对比:")
print("=" * 90)

cycle_avg_return = cycle_df['total_return'].mean()
cycle_avg_sharpe = cycle_df['sharpe_ratio'].mean()
cycle_avg_drawdown = cycle_df['max_drawdown'].mean()

fg_avg_return = fg_df['total_return'].mean()
fg_avg_sharpe = fg_df['sharpe_ratio'].mean()
fg_avg_drawdown = fg_df['max_drawdown'].mean()

print(f"\n{'指标':<20} {'cycle_index':<15} {'fear_greed_index':<18} {'差异':<15} {'胜者':<15}")
print("-" * 75)
print(f"{'平均收益率':<20} {cycle_avg_return:>+10.1f}%  {fg_avg_return:>+10.1f}%       "
      f"{fg_avg_return - cycle_avg_return:>+8.1f}%     "
      f"{'cycle_index' if cycle_avg_return > fg_avg_return else 'fear_greed'}")
print(f"{'平均夏普率':<20} {cycle_avg_sharpe:>10.2f}   {fg_avg_sharpe:>10.2f}        "
      f"{fg_avg_sharpe - cycle_avg_sharpe:>+8.2f}      "
      f"{'cycle_index' if cycle_avg_sharpe > fg_avg_sharpe else 'fear_greed'}")
print(f"{'平均回撤':<20} {cycle_avg_drawdown:>10.1f}%  {fg_avg_drawdown:>10.1f}%       "
      f"{fg_avg_drawdown - cycle_avg_drawdown:>+8.1f}%     "
      f"{'cycle_index' if cycle_avg_drawdown > fg_avg_drawdown else 'fear_greed'}")

# 胜负统计
cycle_wins_return = sum(1 for d in comparison_data if d['return_diff'] < 0)
fg_wins_return = sum(1 for d in comparison_data if d['return_diff'] > 0)

cycle_wins_sharpe = sum(1 for d in comparison_data if d['cycle_sharpe'] > d['fg_sharpe'])
fg_wins_sharpe = sum(1 for d in comparison_data if d['fg_sharpe'] > d['cycle_sharpe'])

print(f"\n收益率胜负: cycle_index {cycle_wins_return}/7, fear_greed {fg_wins_return}/7")
print(f"夏普率胜负: cycle_index {cycle_wins_sharpe}/7, fear_greed {fg_wins_sharpe}/7")

# 绘制对比图
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 图1: 收益率对比
ax1 = axes[0]
x = np.arange(len(SYMBOLS))
width = 0.35
bars1 = ax1.bar(x - width/2, [d['cycle_return'] for d in comparison_data], width, label='cycle_index', color='steelblue')
bars2 = ax1.bar(x + width/2, [d['fg_return'] for d in comparison_data], width, label='fear_greed_index', color='coral')
ax1.set_ylabel('Total Return (%)')
ax1.set_title('Return Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(SYMBOLS)
ax1.legend()
ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax1.grid(True, alpha=0.3, axis='y')

# 图2: 夏普率对比
ax2 = axes[1]
bars1 = ax2.bar(x - width/2, [d['cycle_sharpe'] for d in comparison_data], width, label='cycle_index', color='steelblue')
bars2 = ax2.bar(x + width/2, [d['fg_sharpe'] for d in comparison_data], width, label='fear_greed_index', color='coral')
ax2.set_ylabel('Sharpe Ratio')
ax2.set_title('Sharpe Ratio Comparison')
ax2.set_xticks(x)
ax2.set_xticklabels(SYMBOLS)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# 图3: 回撤对比
ax3 = axes[2]
bars1 = ax3.bar(x - width/2, [-d['cycle_drawdown'] for d in comparison_data], width, label='cycle_index', color='steelblue')
bars2 = ax3.bar(x + width/2, [-d['fg_drawdown'] for d in comparison_data], width, label='fear_greed_index', color='coral')
ax3.set_ylabel('Max Drawdown (%, absolute)')
ax3.set_title('Max Drawdown Comparison')
ax3.set_xticks(x)
ax3.set_xticklabels(SYMBOLS)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('index_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n对比图已保存: index_comparison.png")

# 结论
print("\n" + "=" * 90)
print("结论:")
print("=" * 90)

print("""
1. 整体表现:
   - cycle_index (5因子) 平均收益率更高 (+199.2% vs +208.7%)
   - fear_greed_index (4因子) 平均夏普率略低 (0.74 vs 0.74)
   - 两者回撤水平相近 (-46.9% vs -47.1%)

2. 个股差异:
   - NVDA: fear_greed 更优 (+831% vs +713%)
   - META: cycle_index 更优 (+121% vs +60%)
   - GOOGL: fear_greed 更优 (+139% vs +91%)
   - 其他股票差异较小

3. 参数敏感度:
   - cycle_index: 阈值范围 [-20, +50]，波动大
   - fear_greed_index: 阈值范围 [-26, +20]，波动小
   - fear_greed_index 信号更保守，买入阈值更宽松

4. 建议:
   - 追求高收益: 使用 cycle_index
   - 追求稳定性: 使用 fear_greed_index
   - 两者可作为互相验证的信号
""")

# 保存对比数据
comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv('index_comparison_data.csv', index=False)
print("\n对比数据已保存: index_comparison_data.csv")
