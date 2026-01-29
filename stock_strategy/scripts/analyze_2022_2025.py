#!/usr/bin/env python3
"""
重新分析：基于2022-2025数据（Window2-5）给出2026推荐
"""

import pandas as pd
import numpy as np
from pathlib import Path

# 七姐妹股票
symbols = ['MSFT', 'AAPL', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA']
smoothings = ['s3', 's5']

results_dir = Path('/Users/sc2025/Desktop/test/AAPL/sentiment_strategy/results')

def load_latest_results(symbol, smoothing):
    """加载最新的Walk-Forward结果"""
    # 使用walk_forward_{symbol}_{smoothing}_*.csv格式（包含窗口信息）
    pattern = f'walk_forward_{symbol}_{smoothing}_*.csv'
    files = list(results_dir.glob(pattern))
    if not files:
        return None
    latest_file = max(files, key=lambda x: x.stat().st_mtime)
    df = pd.read_csv(latest_file)

    # 重命名列以保持一致
    df = df.rename(columns={
        'test_return': 'annualized_return',
        'test_sharpe': 'sharpe_ratio',
        'test_max_dd': 'max_drawdown',
        'is_profitable': 'profitable',
        'buy': 'buy_threshold',
        'and': 'sell_and_threshold',
        'or': 'sell_or_threshold'
    })

    return df

def calculate_score(row):
    """计算评分（与之前相同的逻辑）"""
    score = 0

    # 1. 盈利能力 (40分)
    if row['profitable']:
        score += 40

    # 2. Sharpe比率 (30分)
    sharpe = row['sharpe_ratio']
    if sharpe >= 1.5:
        score += 30
    elif sharpe >= 1.0:
        score += 25
    elif sharpe >= 0.8:
        score += 20
    elif sharpe >= 0.5:
        score += 10

    # 3. 最大回撤 (20分)
    max_dd = abs(row['max_drawdown'])
    if max_dd <= 10:
        score += 20
    elif max_dd <= 15:
        score += 15
    elif max_dd <= 20:
        score += 10
    elif max_dd <= 30:
        score += 5

    # 4. 收益率 (10分)
    ret = row['annualized_return']
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

def analyze_period(windows_to_include, period_name):
    """分析指定窗口期"""
    print(f"\n{'='*80}")
    print(f"{period_name}分析")
    print(f"{'='*80}\n")

    all_results = []

    for symbol in symbols:
        for smoothing in smoothings:
            df = load_latest_results(symbol, smoothing)
            if df is None:
                continue

            # 只保留指定窗口
            df = df[df['window'].isin(windows_to_include)]

            if len(df) == 0:
                continue

            # 计算每个窗口的评分
            df['score'] = df.apply(calculate_score, axis=1)

            # 汇总统计
            summary = {
                'symbol': symbol,
                'smoothing': smoothing,
                'num_windows': len(df),
                'profitable_windows': df['profitable'].sum(),
                'profitable_rate': df['profitable'].sum() / len(df),
                'avg_return': df['annualized_return'].mean(),
                'avg_sharpe': df['sharpe_ratio'].mean(),
                'avg_drawdown': df['max_drawdown'].mean(),
                'avg_score': df['score'].mean(),
            }

            # 评级
            avg_score = summary['avg_score']
            if avg_score >= 85:
                grade = 'A'
            elif avg_score >= 70:
                grade = 'B'
            else:
                grade = 'C'
            summary['grade'] = grade

            # 保存最新窗口参数（用于2026推荐）
            latest_window = df[df['window'] == df['window'].max()].iloc[0]
            summary['latest_buy'] = latest_window['buy_threshold']
            summary['latest_and'] = latest_window['sell_and_threshold']
            summary['latest_or'] = latest_window['sell_or_threshold']

            all_results.append(summary)

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(['avg_score'], ascending=False)

    return results_df

def main():
    print("="*80)
    print("2022-2025数据分析（Window2-5）")
    print("用户要求：基于2022-2025数据给出2026推荐")
    print("="*80)

    # 分析Window2-5（2022-2025）
    windows_2022_2025 = ['Window2', 'Window3', 'Window4', 'Window5']
    results_2022_2025 = analyze_period(windows_2022_2025, "2022-2025期间（Window2-5）")

    print("\n## 所有股票表现总览（2022-2025）\n")
    print(results_2022_2025[['symbol', 'smoothing', 'grade', 'avg_score',
                              'profitable_rate', 'avg_return', 'avg_sharpe']].to_string(index=False))

    # 分别展示S3和S5的最佳表现
    print("\n\n" + "="*80)
    print("各股票最佳Smoothing选择（2022-2025）")
    print("="*80 + "\n")

    best_by_symbol = results_2022_2025.loc[results_2022_2025.groupby('symbol')['avg_score'].idxmax()]
    best_by_symbol = best_by_symbol.sort_values('avg_score', ascending=False)

    print(best_by_symbol[['symbol', 'smoothing', 'grade', 'avg_score',
                           'profitable_rate', 'avg_return', 'avg_sharpe',
                           'latest_buy', 'latest_and', 'latest_or']].to_string(index=False))

    # 筛选A/B级股票
    print("\n\n" + "="*80)
    print("2026年推荐股票（A/B级）")
    print("="*80 + "\n")

    qualified = best_by_symbol[best_by_symbol['grade'].isin(['A', 'B'])]

    if len(qualified) == 0:
        print("⚠️ 警告：没有股票达到A/B级！")
        print("\n降低标准，展示评分最高的前3只：\n")
        qualified = best_by_symbol.head(3)

    for idx, row in qualified.iterrows():
        print(f"\n### {row['symbol']} - {row['grade']}级 ({int(row['avg_score'])}分)")
        print(f"Smoothing: {row['smoothing'].upper()}")
        print(f"盈利率: {row['profitable_rate']*100:.0f}% ({int(row['profitable_windows'])}/4窗口)")
        print(f"平均年化收益: {row['avg_return']:.2f}%")
        print(f"平均Sharpe: {row['avg_sharpe']:.2f}")
        print(f"平均回撤: {row['avg_drawdown']:.2f}%")
        print(f"\n2026推荐参数:")
        print(f"  buy: < {int(row['latest_buy'])}")
        print(f"  and: > {int(row['latest_and'])}")
        print(f"  or: > {int(row['latest_or'])}")

    # 保存结果
    output_file = results_dir / 'analysis_2022_2025.csv'
    results_2022_2025.to_csv(output_file, index=False)
    print(f"\n\n✅ 完整结果已保存: {output_file}")

    # 对比2023-2025（之前的分析）
    print("\n\n" + "="*80)
    print("对比分析：2022-2025 vs 2023-2025")
    print("="*80 + "\n")

    windows_2023_2025 = ['Window3', 'Window4', 'Window5']
    results_2023_2025 = analyze_period(windows_2023_2025, "2023-2025期间（Window3-5）")
    best_2023_2025 = results_2023_2025.loc[results_2023_2025.groupby('symbol')['avg_score'].idxmax()]

    print("\n对比表：\n")
    comparison = []
    for symbol in symbols:
        row_2022 = best_by_symbol[best_by_symbol['symbol'] == symbol]
        row_2023 = best_2023_2025[best_2023_2025['symbol'] == symbol]

        if len(row_2022) > 0 and len(row_2023) > 0:
            row_2022 = row_2022.iloc[0]
            row_2023 = row_2023.iloc[0]

            comparison.append({
                'symbol': symbol,
                'score_2022_2025': int(row_2022['avg_score']),
                'grade_2022_2025': row_2022['grade'],
                'score_2023_2025': int(row_2023['avg_score']),
                'grade_2023_2025': row_2023['grade'],
                'score_diff': int(row_2022['avg_score'] - row_2023['avg_score']),
            })

    comp_df = pd.DataFrame(comparison)
    comp_df = comp_df.sort_values('score_2022_2025', ascending=False)
    print(comp_df.to_string(index=False))

    print("\n\n" + "="*80)
    print("核心结论")
    print("="*80)
    print("\n包含2022年数据后，推荐是否有变化？")
    print("请查看上述对比表，关注评分和评级的差异。")

if __name__ == '__main__':
    main()
