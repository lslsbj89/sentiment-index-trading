#!/usr/bin/env python3
"""
对比两种评分方法：
1. 旧方法：盈利40% + Sharpe30% + 回撤20% + 收益10%
2. 新方法：收益50% + Sharpe30% + 回撤20%（用户建议）
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

def normalize_score(value, min_val, max_val, reverse=False):
    """
    将指标归一化到0-100

    Args:
        value: 原始值
        min_val: 最小值（对应0分或100分）
        max_val: 最大值（对应100分或0分）
        reverse: True表示越小越好（如回撤），False表示越大越好（如收益、Sharpe）
    """
    if max_val == min_val:
        return 50

    normalized = (value - min_val) / (max_val - min_val) * 100

    if reverse:
        normalized = 100 - normalized

    # 限制在0-100范围
    return max(0, min(100, normalized))

def new_scoring(row, return_range, sharpe_range, dd_range):
    """
    新评分方法（用户建议）

    得分 = 年化收益率(归一化) × 0.5 + Sharpe(归一化) × 0.3 + 回撤率(归一化) × 0.2

    归一化范围：
        - 收益率：-50% ~ 200% → 0-100分
        - Sharpe：-2 ~ 3 → 0-100分
        - 回撤：-50% ~ 0% → 100-0分（越小越好）
    """
    # 归一化年化收益率
    return_score = normalize_score(row['test_return'], return_range[0], return_range[1], reverse=False)

    # 归一化Sharpe
    sharpe_score = normalize_score(row['test_sharpe'], sharpe_range[0], sharpe_range[1], reverse=False)

    # 归一化回撤（越小越好）
    dd_score = normalize_score(row['test_max_dd'], dd_range[0], dd_range[1], reverse=True)

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

    # 确定归一化范围（基于所有数据的分布）
    return_range = (all_df['test_return'].min(), all_df['test_return'].max())
    sharpe_range = (all_df['test_sharpe'].min(), all_df['test_sharpe'].max())
    dd_range = (all_df['test_max_dd'].min(), all_df['test_max_dd'].max())

    print("="*80)
    print("归一化范围（基于2022-2025所有数据）")
    print("="*80)
    print(f"年化收益率范围: {return_range[0]:.2f}% ~ {return_range[1]:.2f}%")
    print(f"Sharpe范围: {sharpe_range[0]:.2f} ~ {sharpe_range[1]:.2f}")
    print(f"最大回撤范围: {dd_range[0]:.2f}% ~ {dd_range[1]:.2f}%")

    # 计算两种评分
    all_df['old_score'] = all_df.apply(old_scoring, axis=1)

    new_scores = all_df.apply(
        lambda row: new_scoring(row, return_range, sharpe_range, dd_range),
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

    print(summary[['symbol', 'smoothing', 'old_avg_score', 'old_grade',
                   'new_avg_score', 'new_grade', 'score_diff', 'rank_change']].to_string(index=False))

    # 详细对比
    print("\n" + "="*80)
    print("评级变化分析")
    print("="*80 + "\n")

    grade_changed = summary[summary['old_grade'] != summary['new_grade']]

    if len(grade_changed) > 0:
        print(f"有 {len(grade_changed)} 个股票的评级发生变化:\n")
        for idx, row in grade_changed.iterrows():
            change = '⬆️ 提升' if row['new_grade'] < row['old_grade'] else '⬇️ 降低'
            print(f"{row['symbol']} ({row['smoothing'].upper()}): {row['old_grade']}级 → {row['new_grade']}级 {change}")
            print(f"  旧评分: {row['old_avg_score']:.1f}, 新评分: {row['new_avg_score']:.1f} (差距: {row['score_diff']:+.1f})")
    else:
        print("✅ 没有股票的评级发生变化")

    # 排名变化
    print("\n" + "="*80)
    print("排名变化分析")
    print("="*80 + "\n")

    rank_changed = summary[abs(summary['rank_change']) > 0].sort_values('rank_change', ascending=False)

    if len(rank_changed) > 0:
        print(f"有 {len(rank_changed)} 个股票的排名发生变化:\n")
        for idx, row in rank_changed.iterrows():
            change = '⬆️ 上升' if row['rank_change'] > 0 else '⬇️ 下降'
            print(f"{row['symbol']} ({row['smoothing'].upper()}): 第{int(row['old_rank'])}名 → 第{int(row['new_rank'])}名 {change} {int(abs(row['rank_change']))}位")
    else:
        print("✅ 排名完全一致")

    # 显示最佳股票对比
    print("\n" + "="*80)
    print("Top 3 对比")
    print("="*80 + "\n")

    old_top3 = summary.nlargest(3, 'old_avg_score')
    new_top3 = summary.nlargest(3, 'new_avg_score')

    print("旧方法 Top 3:")
    for i, (idx, row) in enumerate(old_top3.iterrows(), 1):
        print(f"{i}. {row['symbol']} ({row['smoothing'].upper()}): {row['old_avg_score']:.1f}分 ({row['old_grade']}级)")

    print("\n新方法 Top 3:")
    for i, (idx, row) in enumerate(new_top3.iterrows(), 1):
        print(f"{i}. {row['symbol']} ({row['smoothing'].upper()}): {row['new_avg_score']:.1f}分 ({row['new_grade']}级)")

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
                'old_score': old_row['old_avg_score'],
                'old_grade': old_row['old_grade'],
                'new_smoothing': new_row['smoothing'].upper(),
                'new_score': new_row['new_avg_score'],
                'new_grade': new_row['new_grade'],
                'smoothing_changed': '⚠️' if old_row['smoothing'] != new_row['smoothing'] else '✅'
            })

    comp_df = pd.DataFrame(comparison)
    print(comp_df.to_string(index=False))

    # 保存结果
    output_file = results_dir / 'scoring_comparison.csv'
    summary.to_csv(output_file, index=False)
    print(f"\n✅ 完整对比结果已保存: {output_file}")

    # 总结
    print("\n" + "="*80)
    print("💡 总结与建议")
    print("="*80 + "\n")

    print("新评分方法的优势:")
    print("1. ✅ 收益率是连续指标，更精细地反映表现差异")
    print("2. ✅ 避免了\"盈利1%得40分，亏损1%得0分\"的悬崖效应")
    print("3. ✅ 权重分配合理：收益50% > Sharpe30% > 回撤20%")

    print("\n新评分方法的特点:")
    print("1. ⚠️ 高收益策略得分提升（如TSLA）")
    print("2. ⚠️ 低收益但稳健策略得分可能降低")
    print("3. ⚠️ 需要根据数据范围动态调整归一化参数")

    print("\n建议:")
    if len(grade_changed) == 0 and len(rank_changed) <= 2:
        print("✅ 两种方法结果差异不大，可以切换到新方法")
    else:
        print("⚠️ 两种方法结果有明显差异，建议:")
        print("   - 查看哪些股票的评级/排名发生了变化")
        print("   - 判断新方法是否更符合您的投资目标")
        print("   - 如果追求高收益，用新方法")
        print("   - 如果追求稳健，可能旧方法更合适")

def main():
    print("="*80)
    print("评分方法对比分析")
    print("="*80)
    print("\n旧方法: 盈利40% + Sharpe30% + 回撤20% + 收益10%")
    print("新方法: 收益50% + Sharpe30% + 回撤20%（归一化）\n")

    analyze_all_symbols()

if __name__ == '__main__':
    main()
