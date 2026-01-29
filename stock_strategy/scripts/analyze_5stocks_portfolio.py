#!/usr/bin/env python3
"""
分析5只股票组合：NVDA, TSLA, AAPL, GOOGL, MSFT
基于2022-2025数据给出2026年投资建议
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
    """计算评分（旧方法）"""
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

def analyze_symbol(symbol, smoothings=['s3', 's5']):
    """分析单个股票"""
    windows_2022_2025 = ['Window2', 'Window3', 'Window4', 'Window5']

    results = {}

    for smoothing in smoothings:
        df = load_walk_forward_results(symbol, smoothing)
        if df is None:
            continue

        df = df[df['window'].isin(windows_2022_2025)].copy()
        if len(df) == 0:
            continue

        # 计算评分
        df['score'] = df.apply(calculate_score, axis=1)

        # 汇总
        summary = {
            'symbol': symbol,
            'smoothing': smoothing,
            'avg_score': df['score'].mean(),
            'profitable_rate': df['is_profitable'].sum() / len(df),
            'avg_return': df['test_return'].mean(),
            'avg_sharpe': df['test_sharpe'].mean(),
            'avg_drawdown': df['test_max_dd'].mean(),
            'num_windows': len(df),
            'profitable_windows': df['is_profitable'].sum(),
        }

        # 获取最新窗口参数
        latest = df[df['window'] == 'Window5'].iloc[0]
        summary['latest_buy'] = int(latest['buy'])
        summary['latest_and'] = int(latest['and'])
        summary['latest_or'] = int(latest['or'])

        # 评级
        if summary['avg_score'] >= 85:
            summary['grade'] = 'A'
        elif summary['avg_score'] >= 70:
            summary['grade'] = 'B'
        else:
            summary['grade'] = 'C'

        results[smoothing] = summary

    # 选择最佳smoothing
    if not results:
        return None

    best_smoothing = max(results.keys(), key=lambda s: results[s]['avg_score'])

    return results[best_smoothing]

def main():
    symbols = ['MSFT', 'NVDA', 'TSLA', 'AAPL', 'GOOGL']

    print("="*80)
    print("5只股票组合分析（2022-2025数据）")
    print("股票池: MSFT, NVDA, TSLA, AAPL, GOOGL")
    print("="*80 + "\n")

    # 分析每只股票
    all_results = []

    for symbol in symbols:
        result = analyze_symbol(symbol)
        if result:
            all_results.append(result)

    if not all_results:
        print("❌ 没有数据")
        return

    # 创建DataFrame
    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values('avg_score', ascending=False)

    # 显示总览
    print("## 各股票表现总览（最佳Smoothing）\n")
    print(df_results[['symbol', 'smoothing', 'grade', 'avg_score', 'profitable_rate',
                      'avg_return', 'avg_sharpe', 'avg_drawdown']].to_string(index=False))

    # 详细展示
    print("\n\n" + "="*80)
    print("各股票详细信息与2026推荐参数")
    print("="*80)

    for idx, row in df_results.iterrows():
        print(f"\n### {row['symbol']} - {row['grade']}级 ({int(row['avg_score'])}分)")
        print(f"{'='*70}")

        print(f"\n**Smoothing**: {row['smoothing'].upper()}")

        print(f"\n**2022-2025表现**:")
        print(f"  盈利率: {row['profitable_rate']*100:.0f}% ({int(row['profitable_windows'])}/{int(row['num_windows'])}窗口)")
        print(f"  平均年化收益: {row['avg_return']:.2f}%")
        print(f"  平均Sharpe: {row['avg_sharpe']:.2f}")
        print(f"  平均回撤: {row['avg_drawdown']:.2f}%")

        print(f"\n**2026推荐参数**:")
        print(f"  buy: < {row['latest_buy']}")
        print(f"  and: > {row['latest_and']}")
        print(f"  or:  > {row['latest_or']}")

        # 评价
        if row['grade'] == 'A':
            comment = "⭐⭐⭐ 优秀，强烈推荐"
        elif row['grade'] == 'B':
            comment = "⭐⭐ 良好，推荐"
        else:
            if row['avg_score'] >= 60:
                comment = "⭐ 合格，可配置"
            else:
                comment = "⚠️ 表现一般，谨慎配置"

        print(f"\n**评价**: {comment}")

    # 组合建议
    print("\n\n" + "="*80)
    print("投资组合建议（2026）")
    print("="*80 + "\n")

    # 方案A: 评分加权
    total_score = df_results['avg_score'].sum()
    df_results['weight_by_score'] = df_results['avg_score'] / total_score

    print("### 方案A: 评分加权型 ⭐⭐⭐\n")
    print("根据各股票评分分配仓位:\n")
    for idx, row in df_results.iterrows():
        weight = row['weight_by_score'] * 100
        print(f"{row['symbol']}: {weight:.1f}% ({row['smoothing'].upper()}: buy<{row['latest_buy']}, and>{row['latest_and']}, or>{row['latest_or']})")

    avg_return_weighted = (df_results['avg_return'] * df_results['weight_by_score']).sum()
    avg_sharpe_weighted = (df_results['avg_sharpe'] * df_results['weight_by_score']).sum()

    print(f"\n预期组合表现:")
    print(f"  年化收益: {avg_return_weighted:.2f}%")
    print(f"  Sharpe: {avg_sharpe_weighted:.2f}")
    print(f"  说明: 高评分股票获得更多仓位")

    # 方案B: 等权重
    print("\n\n### 方案B: 等权重型\n")
    print("5只股票各占20%:\n")
    for idx, row in df_results.iterrows():
        print(f"{row['symbol']}: 20% ({row['smoothing'].upper()}: buy<{row['latest_buy']}, and>{row['latest_and']}, or>{row['latest_or']})")

    avg_return_equal = df_results['avg_return'].mean()
    avg_sharpe_equal = df_results['avg_sharpe'].mean()

    print(f"\n预期组合表现:")
    print(f"  年化收益: {avg_return_equal:.2f}%")
    print(f"  Sharpe: {avg_sharpe_equal:.2f}")
    print(f"  说明: 最简单，适合懒人")

    # 方案C: 分层配置
    print("\n\n### 方案C: 分层配置型 ⭐⭐⭐\n")
    print("根据评级分层配置:\n")

    tier_a = df_results[df_results['grade'] == 'A']
    tier_b = df_results[df_results['grade'] == 'B']
    tier_c = df_results[df_results['grade'] == 'C']

    if len(tier_a) > 0:
        print(f"**核心层（A级）**: 占50%")
        a_weight = 50 / len(tier_a)
        for idx, row in tier_a.iterrows():
            print(f"  {row['symbol']}: {a_weight:.1f}% ({row['smoothing'].upper()}: buy<{row['latest_buy']}, and>{row['latest_and']}, or>{row['latest_or']})")

    if len(tier_b) > 0:
        print(f"\n**配置层（B级）**: 占30%")
        b_weight = 30 / len(tier_b)
        for idx, row in tier_b.iterrows():
            print(f"  {row['symbol']}: {b_weight:.1f}% ({row['smoothing'].upper()}: buy<{row['latest_buy']}, and>{row['latest_and']}, or>{row['latest_or']})")

    if len(tier_c) > 0:
        print(f"\n**卫星层（C级）**: 占20%")
        c_weight = 20 / len(tier_c)
        for idx, row in tier_c.iterrows():
            print(f"  {row['symbol']}: {c_weight:.1f}% ({row['smoothing'].upper()}: buy<{row['latest_buy']}, and>{row['latest_and']}, or>{row['latest_or']})")

    print(f"\n说明: 高评级核心持有，低评级小仓位博弈")

    # 方案D: 用户自定义建议
    print("\n\n### 方案D: 稳健型（推荐）⭐⭐⭐\n")
    print("手动优化配置，兼顾收益与风险:\n")

    # 按评分排序，手动分配合理仓位
    if len(df_results) >= 5:
        top1 = df_results.iloc[0]
        top2 = df_results.iloc[1]
        top3 = df_results.iloc[2]
        top4 = df_results.iloc[3]
        top5 = df_results.iloc[4]

        print(f"{top1['symbol']}: 30% (最高分，核心)")
        print(f"{top2['symbol']}: 25% (次高分，主力)")
        print(f"{top3['symbol']}: 20% (第三，平衡)")
        print(f"{top4['symbol']}: 15% (第四，补充)")
        print(f"{top5['symbol']}: 10% (最低分，卫星)")

        custom_return = (top1['avg_return']*0.30 + top2['avg_return']*0.25 +
                        top3['avg_return']*0.20 + top4['avg_return']*0.15 +
                        top5['avg_return']*0.10)
        custom_sharpe = (top1['avg_sharpe']*0.30 + top2['avg_sharpe']*0.25 +
                        top3['avg_sharpe']*0.20 + top4['avg_sharpe']*0.15 +
                        top5['avg_sharpe']*0.10)

        print(f"\n预期组合表现:")
        print(f"  年化收益: {custom_return:.2f}%")
        print(f"  Sharpe: {custom_sharpe:.2f}")
        print(f"  说明: 根据评分梯度分配，收益与风险平衡")

    # 风险提示
    print("\n\n" + "="*80)
    print("⚠️ 风险提示")
    print("="*80 + "\n")

    c_stocks = df_results[df_results['grade'] == 'C']
    if len(c_stocks) > 0:
        print(f"1. **C级股票**: {len(c_stocks)}只")
        for idx, row in c_stocks.iterrows():
            print(f"   - {row['symbol']}: {int(row['avg_score'])}分，盈利率{row['profitable_rate']*100:.0f}%")
        print(f"   ⚠️ C级表示2022-2025表现不稳定，建议小仓位或观察")

    low_sharpe = df_results[df_results['avg_sharpe'] < 1.0]
    if len(low_sharpe) > 0:
        print(f"\n2. **Sharpe<1.0**: {len(low_sharpe)}只")
        for idx, row in low_sharpe.iterrows():
            print(f"   - {row['symbol']}: Sharpe {row['avg_sharpe']:.2f}")
        print(f"   ⚠️ Sharpe<1.0表示风险调整后收益偏低")

    low_profit = df_results[df_results['profitable_rate'] < 1.0]
    if len(low_profit) > 0:
        print(f"\n3. **盈利率<100%**: {len(low_profit)}只")
        for idx, row in low_profit.iterrows():
            print(f"   - {row['symbol']}: {row['profitable_rate']*100:.0f}% ({int(row['profitable_windows'])}/{int(row['num_windows'])}窗口)")
        print(f"   ⚠️ 存在亏损窗口，需要做好心理准备")

    # 实盘建议
    print("\n\n" + "="*80)
    print("💡 实盘建议")
    print("="*80 + "\n")

    print("**推荐方案**: 方案D（稳健型）")
    print("\n**理由**:")
    print("1. 按评分梯度分配，高分股票高仓位")
    print("2. 即使低分股票也保留10-15%，保持分散")
    print("3. 收益与风险平衡")

    print("\n**Smoothing配置**:")
    for idx, row in df_results.iterrows():
        print(f"  {row['symbol']}: {row['smoothing'].upper()}")

    different_smoothing = len(df_results['smoothing'].unique()) > 1
    if different_smoothing:
        print(f"\n  ⚠️ 注意：包含{len(df_results['smoothing'].unique())}种Smoothing参数")
        print(f"  需要在系统中分别计算S3和S5的情绪指数")

    print("\n**风控建议**:")
    print("1. 单只股票止损: 亏损>10% → 平仓")
    print("2. 组合止损: 总亏损>15% → 减仓50%")
    print("3. 复盘频率: 每季度检查")
    print("4. 调仓条件: 某只股票连续2季度亏损 → 减仓或移除")

    # 保存结果
    output_file = results_dir / 'portfolio_5stocks_2026.csv'
    df_results.to_csv(output_file, index=False)
    print(f"\n✅ 结果已保存: {output_file}")

    return df_results

if __name__ == '__main__':
    main()
