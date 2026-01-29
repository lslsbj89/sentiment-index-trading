#!/usr/bin/env python3
"""
改进的评分方法对比
修正归一化问题，使用固定的合理区间而非数据min-max
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

def old_scoring(row):
    """旧评分方法"""
    score = 0

    # 1. 盈利能力 (40分)
    if row['test_return'] > 0:
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

def new_scoring_v2(row):
    """
    改进的新评分方法

    使用固定的合理区间归一化，而非数据min-max：
    - 年化收益率：-50% ~ 100% → 0-100分
    - Sharpe：-1 ~ 3 → 0-100分
    - 回撤：-50% ~ 0% → 100-0分

    得分 = 收益率(归一化) × 0.5 + Sharpe(归一化) × 0.3 + 回撤(归一化) × 0.2
    """
    # 1. 年化收益率归一化 (-50% ~ 100%)
    ret = row['test_return']
    if ret <= -50:
        return_score = 0
    elif ret >= 100:
        return_score = 100
    else:
        return_score = (ret + 50) / 150 * 100

    # 2. Sharpe归一化 (-1 ~ 3)
    sharpe = row['test_sharpe']
    if sharpe <= -1:
        sharpe_score = 0
    elif sharpe >= 3:
        sharpe_score = 100
    else:
        sharpe_score = (sharpe + 1) / 4 * 100

    # 3. 回撤归一化 (-50% ~ 0%)，越小越好
    dd = row['test_max_dd']
    if dd <= -50:
        dd_score = 0
    elif dd >= 0:
        dd_score = 100
    else:
        dd_score = (dd + 50) / 50 * 100

    # 加权求和
    total_score = return_score * 0.5 + sharpe_score * 0.3 + dd_score * 0.2

    return total_score, return_score, sharpe_score, dd_score

def analyze_all_symbols():
    """分析所有七姐妹股票"""
    symbols = ['MSFT', 'AAPL', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA']
    smoothings = ['s3', 's5']
    windows_2022_2025 = ['Window2', 'Window3', 'Window4', 'Window5']

    all_data = []

    # 收集所有数据
    for symbol in symbols:
        for smoothing in smoothings:
            df = load_walk_forward_results(symbol, smoothing)
            if df is None:
                continue

            df = df[df['window'].isin(windows_2022_2025)].copy()
            if len(df) == 0:
                continue

            df['symbol'] = symbol
            df['smoothing'] = smoothing
            all_data.append(df)

    if not all_data:
        print("❌ 没有数据")
        return

    # 合并所有数据
    all_df = pd.concat(all_data, ignore_index=True)

    print("="*80)
    print("改进的新评分方法")
    print("="*80)
    print("\n归一化固定区间:")
    print("  年化收益率: -50% ~ 100% → 0-100分")
    print("  Sharpe比率: -1 ~ 3 → 0-100分")
    print("  最大回撤: -50% ~ 0% → 100-0分（越小越好）")
    print("\n权重分配:")
    print("  收益率: 50%")
    print("  Sharpe: 30%")
    print("  回撤: 20%")

    # 计算两种评分
    all_df['old_score'] = all_df.apply(old_scoring, axis=1)

    new_scores = all_df.apply(
        lambda row: new_scoring_v2(row),
        axis=1,
        result_type='expand'
    )
    all_df['new_score'] = new_scores[0]
    all_df['new_return_score'] = new_scores[1]
    all_df['new_sharpe_score'] = new_scores[2]
    all_df['new_dd_score'] = new_scores[3]

    # 按股票+smoothing汇总
    print("\n" + "="*80)
    print("两种评分方法对比（各股票平均）")
    print("="*80 + "\n")

    summary = all_df.groupby(['symbol', 'smoothing']).agg({
        'test_return': 'mean',
        'test_sharpe': 'mean',
        'test_max_dd': 'mean',
        'is_profitable': lambda x: x.sum() / len(x),
        'old_score': 'mean',
        'new_score': 'mean'
    }).reset_index()

    summary.columns = ['symbol', 'smoothing', 'avg_return', 'avg_sharpe', 'avg_dd',
                       'profit_rate', 'old_avg_score', 'new_avg_score']

    summary['score_diff'] = summary['new_avg_score'] - summary['old_avg_score']
    summary['old_rank'] = summary['old_avg_score'].rank(ascending=False)
    summary['new_rank'] = summary['new_avg_score'].rank(ascending=False)
    summary['rank_change'] = summary['old_rank'] - summary['new_rank']

    # 添加评级
    def get_grade(score):
        if score >= 85:
            return 'A'
        elif score >= 70:
            return 'B'
        else:
            return 'C'

    summary['old_grade'] = summary['old_avg_score'].apply(get_grade)
    summary['new_grade'] = summary['new_avg_score'].apply(get_grade)

    summary = summary.sort_values('new_avg_score', ascending=False)

    print(summary[['symbol', 'smoothing', 'avg_return', 'avg_sharpe', 'profit_rate',
                   'old_avg_score', 'old_grade', 'new_avg_score', 'new_grade', 'score_diff']].to_string(index=False))

    # 详细对比
    print("\n" + "="*80)
    print("评级变化分析")
    print("="*80 + "\n")

    grade_changed = summary[summary['old_grade'] != summary['new_grade']]

    if len(grade_changed) > 0:
        print(f"有 {len(grade_changed)} 个股票的评级发生变化:\n")
        for idx, row in grade_changed.iterrows():
            if row['new_grade'] == 'A' and row['old_grade'] != 'A':
                emoji = '⬆️ 提升至A级'
            elif row['new_grade'] == 'B' and row['old_grade'] == 'C':
                emoji = '⬆️ 提升至B级'
            elif row['old_grade'] == 'A' and row['new_grade'] != 'A':
                emoji = '⬇️ 从A级降低'
            elif row['old_grade'] == 'B' and row['new_grade'] == 'C':
                emoji = '⬇️ 从B级降低'
            else:
                emoji = ''

            print(f"{row['symbol']} ({row['smoothing'].upper()}): {row['old_grade']}级 → {row['new_grade']}级 {emoji}")
            print(f"  旧评分: {row['old_avg_score']:.1f}, 新评分: {row['new_avg_score']:.1f} (差距: {row['score_diff']:+.1f})")
            print(f"  收益: {row['avg_return']:.1f}%, Sharpe: {row['avg_sharpe']:.2f}, 盈利率: {row['profit_rate']*100:.0f}%")
            print()
    else:
        print("✅ 没有股票的评级发生变化")

    # Top 3 对比
    print("\n" + "="*80)
    print("Top 3 对比")
    print("="*80 + "\n")

    old_top3 = summary.nlargest(3, 'old_avg_score')
    new_top3 = summary.nlargest(3, 'new_avg_score')

    print("旧方法 Top 3:")
    for i, (idx, row) in enumerate(old_top3.iterrows(), 1):
        print(f"{i}. {row['symbol']} ({row['smoothing'].upper()}): {row['old_avg_score']:.1f}分 ({row['old_grade']}级) - 收益{row['avg_return']:.1f}%, Sharpe{row['avg_sharpe']:.2f}")

    print("\n新方法 Top 3:")
    for i, (idx, row) in enumerate(new_top3.iterrows(), 1):
        print(f"{i}. {row['symbol']} ({row['smoothing'].upper()}): {row['new_avg_score']:.1f}分 ({row['new_grade']}级) - 收益{row['avg_return']:.1f}%, Sharpe{row['avg_sharpe']:.2f}")

    # 各股票最佳smoothing对比
    print("\n" + "="*80)
    print("各股票最佳Smoothing选择对比")
    print("="*80 + "\n")

    old_best = summary.loc[summary.groupby('symbol')['old_avg_score'].idxmax()].sort_values('old_avg_score', ascending=False)
    new_best = summary.loc[summary.groupby('symbol')['new_avg_score'].idxmax()].sort_values('new_avg_score', ascending=False)

    comparison = []
    for symbol in symbols:
        old_row = old_best[old_best['symbol'] == symbol]
        new_row = new_best[new_best['symbol'] == symbol]

        if len(old_row) > 0 and len(new_row) > 0:
            old_row = old_row.iloc[0]
            new_row = new_row.iloc[0]

            comparison.append({
                'symbol': symbol,
                'old_smoothing': old_row['smoothing'].upper(),
                'old_score': f"{old_row['old_avg_score']:.0f}({old_row['old_grade']})",
                'new_smoothing': new_row['smoothing'].upper(),
                'new_score': f"{new_row['new_avg_score']:.0f}({new_row['new_grade']})",
                'changed': '⚠️' if old_row['smoothing'] != new_row['smoothing'] else '✅'
            })

    comp_df = pd.DataFrame(comparison)
    print(comp_df.to_string(index=False))

    # 保存结果
    output_file = results_dir / 'scoring_comparison_v2.csv'
    summary.to_csv(output_file, index=False)
    print(f"\n✅ 完整对比结果已保存: {output_file}")

    # 总结
    print("\n" + "="*80)
    print("💡 结论与建议")
    print("="*80 + "\n")

    grade_upgraded = grade_changed[grade_changed['new_grade'] < grade_changed['old_grade']]
    grade_downgraded = grade_changed[grade_changed['new_grade'] > grade_changed['old_grade']]

    print(f"评级提升: {len(grade_upgraded)} 个")
    print(f"评级降低: {len(grade_downgraded)} 个")
    print(f"评级不变: {len(summary) - len(grade_changed)} 个")

    print("\n新方法的特点:")
    print("1. ✅ 消除\"盈利1%得40分，亏损1%得0分\"的悬崖效应")
    print("2. ✅ 收益率作为连续指标，更精细反映表现")
    print("3. ✅ 高收益高Sharpe的策略得分提升（如TSLA S5）")
    print("4. ⚠️ 低收益但稳健的策略得分可能略降（如MSFT）")

    print("\n建议:")
    if abs(summary['score_diff'].mean()) < 10:
        print("✅ 两种方法结果接近，可以切换到新方法")
    else:
        print("您的投资目标是什么？")
        print("  - 追求高收益（能承受高波动）→ 用新方法，TSLA/NVDA排名更高")
        print("  - 追求稳健（低回撤）→ 用旧方法，MSFT/AMZN排名更高")
        print("  - 我的建议：两种方法结合使用")
        print("    - 用新方法筛选高潜力股票")
        print("    - 用旧方法验证稳健性")
        print("    - 选择两个方法都表现好的股票")

def main():
    print("="*80)
    print("评分方法对比分析 v2（修正归一化）")
    print("="*80)
    print("\n旧方法: 盈利40% + Sharpe30% + 回撤20% + 收益10%")
    print("新方法: 收益50%(归一化) + Sharpe30%(归一化) + 回撤20%(归一化)\n")

    analyze_all_symbols()

if __name__ == '__main__':
    main()
