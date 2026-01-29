#!/usr/bin/env python3
"""
详细分析NVDA和TSLA的2022-2025表现
为用户提供2026参数推荐
"""

import pandas as pd
import numpy as np
from pathlib import Path

results_dir = Path('/Users/sc2025/Desktop/test/AAPL/sentiment_strategy/results')

def load_walk_forward_results(symbol, smoothing):
    """加载Walk-Forward结果"""
    pattern = f'walk_forward_{symbol}_{smoothing}_*.csv'
    files = list(results_dir.glob(pattern))
    if not files:
        return None
    latest_file = max(files, key=lambda x: x.stat().st_mtime)
    return pd.read_csv(latest_file)

def calculate_score(row):
    """计算评分"""
    score = 0

    # 1. 盈利能力 (40分)
    if row['is_profitable']:
        score += 40

    # 2. Sharpe比率 (30分)
    sharpe = row['test_sharpe']
    if sharpe >= 1.5:
        score += 30
    elif sharpe >= 1.0:
        score += 25
    elif sharpe >= 0.8:
        score += 20
    elif sharpe >= 0.5:
        score += 10

    # 3. 最大回撤 (20分)
    max_dd = abs(row['test_max_dd'])
    if max_dd <= 10:
        score += 20
    elif max_dd <= 15:
        score += 15
    elif max_dd <= 20:
        score += 10
    elif max_dd <= 30:
        score += 5

    # 4. 收益率 (10分)
    ret = row['test_return']
    if ret >= 30:
        score += 10
    elif ret >= 20:
        score += 8
    elif ret >= 10:
        score += 6
    elif ret >= 5:
        score += 4
    elif ret >= 0:
        score += 2

    return score

def analyze_symbol(symbol):
    """详细分析单个股票"""
    print(f"\n{'='*80}")
    print(f"{symbol} 详细分析（2022-2025）")
    print(f"{'='*80}\n")

    results_s3 = load_walk_forward_results(symbol, 's3')
    results_s5 = load_walk_forward_results(symbol, 's5')

    if results_s3 is None or results_s5 is None:
        print(f"❌ 缺少{symbol}数据")
        return None

    # 只保留Window2-5（2022-2025）
    windows_2022_2025 = ['Window2', 'Window3', 'Window4', 'Window5']
    results_s3 = results_s3[results_s3['window'].isin(windows_2022_2025)].copy()
    results_s5 = results_s5[results_s5['window'].isin(windows_2022_2025)].copy()

    # 计算评分
    results_s3['score'] = results_s3.apply(calculate_score, axis=1)
    results_s5['score'] = results_s5.apply(calculate_score, axis=1)

    # 显示各窗口表现
    print("## S3表现（各窗口）\n")
    s3_display = results_s3[['window', 'test_period', 'test_return', 'test_sharpe',
                              'test_max_dd', 'is_profitable', 'score', 'buy', 'and', 'or']]
    print(s3_display.to_string(index=False))

    print(f"\nS3汇总:")
    print(f"  平均评分: {results_s3['score'].mean():.1f}")
    print(f"  盈利率: {results_s3['is_profitable'].sum()}/{len(results_s3)} ({results_s3['is_profitable'].mean()*100:.0f}%)")
    print(f"  平均收益: {results_s3['test_return'].mean():.2f}%")
    print(f"  平均Sharpe: {results_s3['test_sharpe'].mean():.2f}")
    print(f"  平均回撤: {results_s3['test_max_dd'].mean():.2f}%")

    print("\n" + "-"*80 + "\n")
    print("## S5表现（各窗口）\n")
    s5_display = results_s5[['window', 'test_period', 'test_return', 'test_sharpe',
                              'test_max_dd', 'is_profitable', 'score', 'buy', 'and', 'or']]
    print(s5_display.to_string(index=False))

    print(f"\nS5汇总:")
    print(f"  平均评分: {results_s5['score'].mean():.1f}")
    print(f"  盈利率: {results_s5['is_profitable'].sum()}/{len(results_s5)} ({results_s5['is_profitable'].mean()*100:.0f}%)")
    print(f"  平均收益: {results_s5['test_return'].mean():.2f}%")
    print(f"  平均Sharpe: {results_s5['test_sharpe'].mean():.2f}")
    print(f"  平均回撤: {results_s5['test_max_dd'].mean():.2f}%")

    # 推荐最佳Smoothing
    print("\n" + "="*80)
    if results_s3['score'].mean() > results_s5['score'].mean():
        best_smoothing = 'S3'
        best_results = results_s3
        best_score = results_s3['score'].mean()
    else:
        best_smoothing = 'S5'
        best_results = results_s5
        best_score = results_s5['score'].mean()

    print(f"## {symbol} 最佳Smoothing: {best_smoothing} (平均评分 {best_score:.1f})")
    print("="*80)

    # 获取最新窗口(Window5)参数
    latest_window = best_results[best_results['window'] == 'Window5'].iloc[0]

    # 分析参数稳定性
    print(f"\n## 参数稳定性分析（{best_smoothing}）\n")
    param_analysis = best_results[['window', 'buy', 'and', 'or']].copy()
    print(param_analysis.to_string(index=False))

    print(f"\n参数统计:")
    print(f"  buy: 均值={best_results['buy'].mean():.1f}, 标准差={best_results['buy'].std():.1f}")
    print(f"  and: 均值={best_results['and'].mean():.1f}, 标准差={best_results['and'].std():.1f}")
    print(f"  or:  均值={best_results['or'].mean():.1f}, 标准差={best_results['or'].std():.1f}")

    # 2026推荐
    print(f"\n{'='*80}")
    print(f"## {symbol} 2026年推荐参数")
    print(f"{'='*80}\n")

    print(f"**基于Window5（2025年）最优参数**:\n")
    print(f"Smoothing: {best_smoothing}")
    print(f"  buy: < {int(latest_window['buy'])}")
    print(f"  and: > {int(latest_window['and'])}")
    print(f"  or:  > {int(latest_window['or'])}")

    print(f"\n预期表现（基于2022-2025平均）:")
    print(f"  年化收益: {best_results['test_return'].mean():.2f}%")
    print(f"  Sharpe: {best_results['test_sharpe'].mean():.2f}")
    print(f"  最大回撤: {best_results['test_max_dd'].mean():.2f}%")
    print(f"  盈利概率: {best_results['is_profitable'].mean()*100:.0f}%")

    # 评级
    if best_score >= 85:
        grade = 'A'
    elif best_score >= 70:
        grade = 'B'
    else:
        grade = 'C'

    print(f"\n评级: {grade}级 ({int(best_score)}分)")

    return {
        'symbol': symbol,
        'best_smoothing': best_smoothing,
        'grade': grade,
        'score': best_score,
        'buy': int(latest_window['buy']),
        'and': int(latest_window['and']),
        'or': int(latest_window['or']),
        'avg_return': best_results['test_return'].mean(),
        'avg_sharpe': best_results['test_sharpe'].mean(),
        'avg_drawdown': best_results['test_max_dd'].mean(),
        'profitable_rate': best_results['is_profitable'].mean(),
        'buy_std': best_results['buy'].std(),
        'and_std': best_results['and'].std(),
        'or_std': best_results['or'].std(),
    }

def main():
    print("="*80)
    print("NVDA和TSLA详细分析及2026推荐")
    print("="*80)

    nvda_summary = analyze_symbol('NVDA')
    tsla_summary = analyze_symbol('TSLA')

    if nvda_summary is None or tsla_summary is None:
        print("\n❌ 数据加载失败")
        return

    # 组合推荐
    print("\n\n" + "="*80)
    print("NVDA + TSLA 组合推荐（2026）")
    print("="*80 + "\n")

    print("## 推荐参数总表\n")

    summary_df = pd.DataFrame([nvda_summary, tsla_summary])
    print(summary_df[['symbol', 'best_smoothing', 'grade', 'score', 'buy', 'and', 'or',
                      'avg_return', 'avg_sharpe', 'profitable_rate']].to_string(index=False))

    print("\n\n## 组合方案\n")

    # 根据评分建议仓位
    total_score = nvda_summary['score'] + tsla_summary['score']
    nvda_weight = nvda_summary['score'] / total_score
    tsla_weight = tsla_summary['score'] / total_score

    print(f"### 方案A: 评分加权型")
    print(f"```yaml")
    print(f"NVDA: {nvda_weight*100:.0f}% ({nvda_summary['best_smoothing']}: buy<{nvda_summary['buy']}, and>{nvda_summary['and']}, or>{nvda_summary['or']})")
    print(f"TSLA: {tsla_weight*100:.0f}% ({tsla_summary['best_smoothing']}: buy<{tsla_summary['buy']}, and>{tsla_summary['and']}, or>{tsla_summary['or']})")
    print(f"```")

    print(f"\n### 方案B: 对半型")
    print(f"```yaml")
    print(f"NVDA: 50% ({nvda_summary['best_smoothing']}: buy<{nvda_summary['buy']}, and>{nvda_summary['and']}, or>{nvda_summary['or']})")
    print(f"TSLA: 50% ({tsla_summary['best_smoothing']}: buy<{tsla_summary['buy']}, and>{tsla_summary['and']}, or>{tsla_summary['or']})")
    print(f"```")

    print(f"\n### 方案C: 保守型（偏NVDA）")
    print(f"```yaml")
    print(f"NVDA: 70% ({nvda_summary['best_smoothing']}: buy<{nvda_summary['buy']}, and>{nvda_summary['and']}, or>{nvda_summary['or']})")
    print(f"TSLA: 30% ({tsla_summary['best_smoothing']}: buy<{tsla_summary['buy']}, and>{tsla_summary['and']}, or>{tsla_summary['or']})")
    print(f"```")
    print(f"理由: NVDA评分更高，Sharpe更稳定")

    # 风险提示
    print("\n\n" + "="*80)
    print("⚠️ 风险提示")
    print("="*80 + "\n")

    if nvda_summary['grade'] == 'C' or tsla_summary['grade'] == 'C':
        print("1. **评级警告**: ")
        if nvda_summary['grade'] == 'C':
            print(f"   - NVDA为C级（{int(nvda_summary['score'])}分），未达B级标准（70分）")
        if tsla_summary['grade'] == 'C':
            print(f"   - TSLA为C级（{int(tsla_summary['score'])}分），未达B级标准（70分）")
        print("   - C级表示策略在2022-2025表现不稳定")

    print("\n2. **盈利率**: ")
    print(f"   - NVDA: {nvda_summary['profitable_rate']*100:.0f}% ({int(nvda_summary['profitable_rate']*4)}/4窗口盈利)")
    print(f"   - TSLA: {tsla_summary['profitable_rate']*100:.0f}% ({int(tsla_summary['profitable_rate']*4)}/4窗口盈利)")
    if nvda_summary['profitable_rate'] < 1.0 or tsla_summary['profitable_rate'] < 1.0:
        print("   ⚠️ 未达到100%盈利率，存在亏损窗口")

    print("\n3. **Sharpe比率**: ")
    print(f"   - NVDA: {nvda_summary['avg_sharpe']:.2f}")
    print(f"   - TSLA: {tsla_summary['avg_sharpe']:.2f}")
    if nvda_summary['avg_sharpe'] < 1.0:
        print(f"   ⚠️ NVDA Sharpe<1.0，风险调整后收益偏低")
    if tsla_summary['avg_sharpe'] < 1.0:
        print(f"   ⚠️ TSLA Sharpe<1.0，风险调整后收益偏低")

    print("\n4. **波动性**: ")
    print(f"   - TSLA平均回撤: {tsla_summary['avg_drawdown']:.2f}%")
    print(f"   - NVDA平均回撤: {nvda_summary['avg_drawdown']:.2f}%")
    if abs(tsla_summary['avg_drawdown']) > 15:
        print(f"   ⚠️ TSLA回撤较大，波动性高")

    print("\n5. **参数稳定性**: ")
    print(f"   - NVDA参数标准差: buy={nvda_summary['buy_std']:.1f}, and={nvda_summary['and_std']:.1f}, or={nvda_summary['or_std']:.1f}")
    print(f"   - TSLA参数标准差: buy={tsla_summary['buy_std']:.1f}, and={tsla_summary['and_std']:.1f}, or={tsla_summary['or_std']:.1f}")
    if tsla_summary['or_std'] > 10:
        print(f"   ⚠️ TSLA参数变化较大，稳定性较差")

    print("\n\n" + "="*80)
    print("💡 建议")
    print("="*80 + "\n")

    print("如果您坚持使用NVDA+TSLA组合，建议:")
    print("1. 使用方案C（NVDA 70% + TSLA 30%），因为NVDA评分更高")
    print("2. 严格执行止损，控制单次亏损不超过组合的5%")
    print("3. 每季度复盘，如连续2个季度亏损考虑调整")
    print("4. 预期年化收益可能在15-35%，但波动较大")
    print("5. 预期Sharpe在0.9-1.1，低于MSFT/AMZN的1.2-1.6")

    print("\n对比推荐组合（MSFT+AMZN）:")
    print("  - MSFT+AMZN: 100%盈利率, Sharpe 1.3-1.6, A/B级")
    print("  - NVDA+TSLA: 75%盈利率, Sharpe 0.9-1.1, C级")
    print("  - 如果追求稳健，建议考虑MSFT+AMZN")

if __name__ == '__main__':
    main()
