#!/usr/bin/env python3
"""
Walk-Forward 情绪指数策略验证

设计：
- 训练期：4年
- 测试期：1年
- 步长：1年
- 避免未来信息泄露，真实验证策略稳健性
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime
from itertools import product
from data_loader import DataLoader

db_config = {
    'host': 'localhost',
    'database': 'crypto_fear_greed_2',
    'user': 'sc2025',
    'password': '',
    'port': 5432
}

def load_sentiment_index(symbol, smoothing=5, start_date='2016-01-01', end_date='2025-12-31'):
    """加载情绪指数"""
    conn = psycopg2.connect(**db_config)
    # smoothing=5 使用 fear_greed_index (3因子指数)
    # smoothing=3 使用 fear_greed_index_s3
    table_name = "fear_greed_index_s3" if smoothing == 3 else "fear_greed_index"
    query = f"""
        SELECT date, smoothed_index
        FROM {table_name}
        WHERE symbol = '{symbol}'
          AND date >= '{start_date}'
          AND date <= '{end_date}'
        ORDER BY date
    """
    df = pd.read_sql(query, conn, parse_dates=['date'])
    conn.close()
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df.set_index('date', inplace=True)
    return df

def backtest_strategy(prices, sentiment, buy_threshold, sell_and_threshold, sell_or_threshold):
    """
    回测策略（已移除60天超时限制）

    卖出条件：
    1. OR卖出：指数 > or_threshold
    2. AND卖出：指数 > and_threshold 且 价格 < MA50
    3. 期末平仓：回测结束仍有持仓
    """
    df = prices.copy()
    df['idx'] = sentiment['smoothed_index']
    df['MA50'] = df['Close'].rolling(50).mean()
    df = df.dropna()

    if len(df) < 100:
        return None

    cash = 100000
    position = 0
    portfolio_values = []
    trades = []

    entry_price = 0
    entry_date = None

    for i in range(len(df)):
        current_date = df.index[i]
        current_price = df['Close'].iloc[i]
        current_idx = df['idx'].iloc[i]
        current_ma50 = df['MA50'].iloc[i]

        # 买入逻辑
        if position == 0 and current_idx < buy_threshold:
            available = cash * 0.8
            shares = int(available / (current_price * 1.002))
            if shares > 0:
                cost = shares * current_price * 1.002
                cash -= cost
                position = shares
                entry_price = current_price * 1.002
                entry_date = current_date

        # 卖出逻辑（移除60天限制）
        elif position > 0:
            sell_signal = False
            exit_reason = None

            # 条件1: OR卖出
            if current_idx > sell_or_threshold:
                sell_signal = True
                exit_reason = 'OR'
            # 条件2: AND卖出
            elif current_idx > sell_and_threshold and current_price < current_ma50:
                sell_signal = True
                exit_reason = 'AND'

            if sell_signal:
                revenue = position * current_price * 0.998
                cash += revenue

                profit = revenue - (position * entry_price)
                profit_pct = (profit / (position * entry_price)) * 100

                trades.append({
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'entry_price': entry_price,
                    'exit_price': current_price * 0.998,
                    'entry_idx': df.loc[entry_date, 'idx'] if entry_date in df.index else None,
                    'exit_idx': current_idx,
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'exit_reason': exit_reason
                })

                position = 0

        total_value = cash + position * current_price
        portfolio_values.append(total_value)

    # 期末强制平仓
    if position > 0:
        final_price = df['Close'].iloc[-1]
        revenue = position * final_price * 0.998
        cash += revenue

        profit = revenue - (position * entry_price)
        profit_pct = (profit / (position * entry_price)) * 100

        trades.append({
            'entry_date': entry_date,
            'exit_date': df.index[-1],
            'entry_price': entry_price,
            'exit_price': final_price * 0.998,
            'entry_idx': df.loc[entry_date, 'idx'] if entry_date in df.index else None,
            'exit_idx': df['idx'].iloc[-1],
            'profit': profit,
            'profit_pct': profit_pct,
            'exit_reason': 'EOD'
        })

    final_value = cash
    total_return = (final_value - 100000) / 100000 * 100

    # 计算指标
    portfolio_series = pd.Series(portfolio_values)
    cummax = portfolio_series.cummax()
    drawdown = (portfolio_series - cummax) / cummax * 100
    max_drawdown = drawdown.min()

    returns = portfolio_series.pct_change().dropna()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

    # 交易统计
    if trades:
        win_trades = [t for t in trades if t['profit'] > 0]
        win_rate = len(win_trades) / len(trades) if trades else 0
    else:
        win_rate = 0

    return {
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'win_rate': win_rate,
        'num_trades': len(trades),
        'final_value': final_value,
        'trades': trades
    }

def score_function(result):
    """
    评分函数（综合版）

    组合考虑：
    - Sharpe率（40%权重）
    - 最大回撤（30%权重）
    - 胜率（20%权重）
    - 交易频率（10%权重，避免过度交易）
    """
    if result is None:
        return -9999

    sharpe = result['sharpe_ratio']
    max_dd = abs(result['max_drawdown'])
    win_rate = result['win_rate']
    num_trades = result['num_trades']

    # 归一化交易次数（假设理想交易次数为10-20次/年）
    trade_score = 1.0
    if num_trades < 5:
        trade_score = 0.5  # 交易太少
    elif num_trades > 50:
        trade_score = 0.5  # 交易太多

    # 综合评分
    score = (
        0.4 * sharpe +
        0.3 * (1 - max_dd / 100) +
        0.2 * win_rate +
        0.1 * trade_score
    )

    return score

def grid_search_train(prices, sentiment, param_space):
    """
    在训练期进行网格搜索

    返回：所有参数组合的评分
    """
    results = []

    total_combinations = (
        len(param_space['buy']) *
        len(param_space['and']) *
        len(param_space['or'])
    )

    count = 0
    for buy, and_t, or_t in product(
        param_space['buy'],
        param_space['and'],
        param_space['or']
    ):
        count += 1
        if count % 10 == 0:
            print(f"  搜索进度: {count}/{total_combinations}", end='\r')

        try:
            result = backtest_strategy(prices, sentiment, buy, and_t, or_t)
            if result:
                score = score_function(result)
                results.append({
                    'buy': buy,
                    'and': and_t,
                    'or': or_t,
                    'score': score,
                    **result
                })
        except:
            pass

    print(f"  搜索完成: {len(results)} 个有效结果")

    return pd.DataFrame(results)

def walk_forward_analysis(symbol, smoothing=3):
    """
    Walk-Forward分析主函数

    窗口设计：
    - 训练期：4年
    - 测试期：1年
    - 步长：1年
    """
    print(f"\n{'='*80}")
    print(f"Walk-Forward 分析: {symbol} (Smoothing={smoothing})")
    print(f"{'='*80}\n")

    # 加载数据
    print("加载数据...")
    loader = DataLoader(db_config)
    prices = loader.load_ohlcv(symbol, '2016-01-01', '2025-12-31')
    sentiment = load_sentiment_index(symbol, smoothing, '2016-01-01', '2025-12-31')
    loader.close()

    # 对齐数据
    common_dates = prices.index.intersection(sentiment.index)
    prices = prices.loc[common_dates]
    sentiment = sentiment.loc[common_dates]

    print(f"  价格数据: {prices.index.min().date()} ~ {prices.index.max().date()} ({len(prices)} 条)")
    print(f"  指数数据: {sentiment.index.min().date()} ~ {sentiment.index.max().date()} ({len(sentiment)} 条)")
    print(f"  指数范围: [{sentiment['smoothed_index'].min():.2f}, {sentiment['smoothed_index'].max():.2f}]")

    # 定义窗口（2021-2025测试期，4年训练+1年测试）
    windows = [
        {'name': 'Window1', 'train': ('2017-01-01', '2020-12-31'), 'test': ('2021-01-01', '2021-12-31')},
        {'name': 'Window2', 'train': ('2018-01-01', '2021-12-31'), 'test': ('2022-01-01', '2022-12-31')},
        {'name': 'Window3', 'train': ('2019-01-01', '2022-12-31'), 'test': ('2023-01-01', '2023-12-31')},
        {'name': 'Window4', 'train': ('2020-01-01', '2023-12-31'), 'test': ('2024-01-01', '2024-12-31')},
        {'name': 'Window5', 'train': ('2021-01-01', '2024-12-31'), 'test': ('2025-01-01', '2025-12-31')},
    ]

    # 参数搜索空间（根据smoothing调整）
    if smoothing == 3:
        param_space = {
            'buy': [-15, -10, -5, 0, 5, 10],   # 6个值
            'and': [10, 15, 20, 25, 30],       # 5个值
            'or': [40, 50, 60, 70]             # 4个值
        }
    else:  # smoothing == 5 (3因子指数，范围约 -45 ~ 109)
        param_space = {
            'buy': [-10, 0, 5, 10, 15, 20],    # 6个值 (恐惧阈值)
            'and': [15, 20, 25, 30, 35],       # 5个值 (AND卖出阈值)
            'or': [50, 60, 70, 80, 90]         # 5个值 (OR卖出阈值)
        }
        # 总共 6×5×5 = 150 个组合

    # Walk-Forward循环
    wf_results = []
    all_train_params = []

    for window in windows:
        print(f"\n{'='*70}")
        print(f"{window['name']}: 训练{window['train'][0][:4]}-{window['train'][1][:4]} → 测试{window['test'][0][:4]}")
        print(f"{'='*70}")

        # 训练期数据
        train_prices = prices.loc[window['train'][0]:window['train'][1]]
        train_sentiment = sentiment.loc[window['train'][0]:window['train'][1]]

        print(f"\n📊 训练期网格搜索...")
        train_results = grid_search_train(train_prices, train_sentiment, param_space)

        # 找到最优参数
        best_train = train_results.sort_values('score', ascending=False).iloc[0]
        best_params = {
            'buy': best_train['buy'],
            'and': best_train['and'],
            'or': best_train['or']
        }
        all_train_params.append(best_params)

        print(f"\n✅ 训练期最优参数:")
        print(f"   buy < {best_params['buy']}, and > {best_params['and']}, or > {best_params['or']}")
        print(f"   训练期Sharpe: {best_train['sharpe_ratio']:.4f}")
        print(f"   训练期收益: {best_train['total_return']:.2f}%")
        print(f"   评分: {best_train['score']:.4f}")

        # 测试期数据
        test_prices = prices.loc[window['test'][0]:window['test'][1]]
        test_sentiment = sentiment.loc[window['test'][0]:window['test'][1]]

        print(f"\n📈 测试期回测（用训练期最优参数）...")
        test_result = backtest_strategy(
            test_prices, test_sentiment,
            best_params['buy'], best_params['and'], best_params['or']
        )

        if test_result:
            print(f"\n✅ 测试期结果:")
            print(f"   收益: {test_result['total_return']:.2f}%")
            print(f"   Sharpe: {test_result['sharpe_ratio']:.4f}")
            print(f"   最大回撤: {test_result['max_drawdown']:.2f}%")
            print(f"   胜率: {test_result['win_rate']*100:.1f}%")
            print(f"   交易次数: {test_result['num_trades']}")

            # 判断是否盈利
            is_profitable = test_result['total_return'] > 0
            print(f"   状态: {'✅ 盈利' if is_profitable else '❌ 亏损'}")

            # 显示详细交易记录
            if test_result['trades']:
                print(f"\n   📋 交易明细:")
                for i, t in enumerate(test_result['trades'], 1):
                    entry_date_str = t['entry_date'].strftime('%Y-%m-%d') if hasattr(t['entry_date'], 'strftime') else str(t['entry_date'])[:10]
                    exit_date_str = t['exit_date'].strftime('%Y-%m-%d') if hasattr(t['exit_date'], 'strftime') else str(t['exit_date'])[:10]
                    print(f"   [{i}] {entry_date_str} → {exit_date_str}")
                    print(f"       买入: ${t['entry_price']:.2f} (指数={t['entry_idx']:.1f})")
                    print(f"       卖出: ${t['exit_price']:.2f} (指数={t['exit_idx']:.1f}) [{t['exit_reason']}]")
                    print(f"       收益: {t['profit_pct']:+.1f}%")

            # 收集交易记录
            for t in test_result['trades']:
                t['window'] = window['name']
                t['test_year'] = window['test'][0][:4]
                t['buy_threshold'] = best_params['buy']
                t['and_threshold'] = best_params['and']
                t['or_threshold'] = best_params['or']

            wf_results.append({
                'window': window['name'],
                'train_period': f"{window['train'][0][:4]}-{window['train'][1][:4]}",
                'test_period': window['test'][0][:4],
                'train_sharpe': best_train['sharpe_ratio'],
                'train_return': best_train['total_return'],
                'test_return': test_result['total_return'],
                'test_sharpe': test_result['sharpe_ratio'],
                'test_max_dd': test_result['max_drawdown'],
                'test_win_rate': test_result['win_rate'],
                'test_trades': test_result['num_trades'],
                'is_profitable': is_profitable,
                'trades': test_result['trades'],
                **best_params
            })
        else:
            print("❌ 测试期数据不足")

    # 汇总所有交易
    all_trades = []
    for r in wf_results:
        if 'trades' in r:
            all_trades.extend(r['trades'])

    return pd.DataFrame(wf_results), all_train_params, param_space, all_trades

def analyze_parameter_robustness(all_params):
    """分析参数稳健性"""
    print(f"\n{'='*80}")
    print("参数稳健性分析")
    print(f"{'='*80}\n")

    df_params = pd.DataFrame(all_params)

    print("📊 各参数统计:")
    print(f"\nbuy阈值:")
    print(f"  均值: {df_params['buy'].mean():.1f}")
    print(f"  标准差: {df_params['buy'].std():.1f}")
    print(f"  范围: [{df_params['buy'].min()}, {df_params['buy'].max()}]")
    print(f"  分布: {df_params['buy'].value_counts().sort_index().to_dict()}")

    print(f"\nand阈值:")
    print(f"  均值: {df_params['and'].mean():.1f}")
    print(f"  标准差: {df_params['and'].std():.1f}")
    print(f"  范围: [{df_params['and'].min()}, {df_params['and'].max()}]")
    print(f"  分布: {df_params['and'].value_counts().sort_index().to_dict()}")

    print(f"\nor阈值:")
    print(f"  均值: {df_params['or'].mean():.1f}")
    print(f"  标准差: {df_params['or'].std():.1f}")
    print(f"  范围: [{df_params['or'].min()}, {df_params['or'].max()}]")
    print(f"  分布: {df_params['or'].value_counts().sort_index().to_dict()}")

    # 计算稳健区（均值±1倍标准差）
    robust_zone = {
        'buy': (df_params['buy'].mean() - df_params['buy'].std(),
                df_params['buy'].mean() + df_params['buy'].std()),
        'and': (df_params['and'].mean() - df_params['and'].std(),
                df_params['and'].mean() + df_params['and'].std()),
        'or': (df_params['or'].mean() - df_params['or'].std(),
               df_params['or'].mean() + df_params['or'].std())
    }

    print(f"\n📍 稳健区（均值±1σ）:")
    print(f"  buy: [{robust_zone['buy'][0]:.1f}, {robust_zone['buy'][1]:.1f}]")
    print(f"  and: [{robust_zone['and'][0]:.1f}, {robust_zone['and'][1]:.1f}]")
    print(f"  or: [{robust_zone['or'][0]:.1f}, {robust_zone['or'][1]:.1f}]")

    # 推荐参数（四舍五入到最近的搜索网格点）
    recommended = {
        'buy': round(df_params['buy'].mean()),
        'and': round(df_params['and'].mean() / 5) * 5,  # 四舍五入到5的倍数
        'or': round(df_params['or'].mean() / 5) * 5
    }

    print(f"\n🎯 推荐实盘参数（稳健区中心）:")
    print(f"  buy < {recommended['buy']}")
    print(f"  and > {recommended['and']}")
    print(f"  or > {recommended['or']}")

    return robust_zone, recommended

def evaluate_strategy(wf_results):
    """策略最终评级"""
    print(f"\n{'='*80}")
    print("策略最终评级")
    print(f"{'='*80}\n")

    # 盈利窗口率
    profitable_rate = wf_results['is_profitable'].sum() / len(wf_results) * 100

    # 平均测试指标
    avg_test_return = wf_results['test_return'].mean()
    avg_test_sharpe = wf_results['test_sharpe'].mean()
    avg_test_dd = wf_results['test_max_dd'].mean()
    avg_test_winrate = wf_results['test_win_rate'].mean()

    print(f"📊 Walk-Forward 总体表现:")
    print(f"  窗口数: {len(wf_results)}")
    print(f"  盈利窗口率: {profitable_rate:.1f}% ({wf_results['is_profitable'].sum()}/{len(wf_results)})")
    print(f"  平均测试收益: {avg_test_return:.2f}%")
    print(f"  平均测试Sharpe: {avg_test_sharpe:.4f}")
    print(f"  平均最大回撤: {avg_test_dd:.2f}%")
    print(f"  平均胜率: {avg_test_winrate*100:.1f}%")

    # 评级逻辑
    print(f"\n🎯 策略评级:")

    grade_points = 0

    # 盈利窗口率（40分）
    if profitable_rate >= 80:
        grade_points += 40
        print(f"  ✅ 盈利窗口率 {profitable_rate:.1f}% (≥80%): +40分")
    elif profitable_rate >= 60:
        grade_points += 30
        print(f"  ⚠️ 盈利窗口率 {profitable_rate:.1f}% (60-80%): +30分")
    else:
        grade_points += 10
        print(f"  ❌ 盈利窗口率 {profitable_rate:.1f}% (<60%): +10分")

    # 平均Sharpe（30分）
    if avg_test_sharpe >= 1.0:
        grade_points += 30
        print(f"  ✅ 平均Sharpe {avg_test_sharpe:.2f} (≥1.0): +30分")
    elif avg_test_sharpe >= 0.8:
        grade_points += 20
        print(f"  ⚠️ 平均Sharpe {avg_test_sharpe:.2f} (0.8-1.0): +20分")
    else:
        grade_points += 10
        print(f"  ❌ 平均Sharpe {avg_test_sharpe:.2f} (<0.8): +10分")

    # 平均回撤（20分）
    if abs(avg_test_dd) <= 15:
        grade_points += 20
        print(f"  ✅ 平均回撤 {avg_test_dd:.1f}% (≤15%): +20分")
    elif abs(avg_test_dd) <= 25:
        grade_points += 10
        print(f"  ⚠️ 平均回撤 {avg_test_dd:.1f}% (15-25%): +10分")
    else:
        grade_points += 0
        print(f"  ❌ 平均回撤 {avg_test_dd:.1f}% (>25%): +0分")

    # 平均收益（10分）
    if avg_test_return >= 20:
        grade_points += 10
        print(f"  ✅ 平均收益 {avg_test_return:.1f}% (≥20%): +10分")
    elif avg_test_return >= 10:
        grade_points += 5
        print(f"  ⚠️ 平均收益 {avg_test_return:.1f}% (10-20%): +5分")
    else:
        grade_points += 0
        print(f"  ❌ 平均收益 {avg_test_return:.1f}% (<10%): +0分")

    print(f"\n总分: {grade_points}/100")

    # 最终评级
    if grade_points >= 85:
        grade = "🟢 A级 - 可实盘"
        recommendation = "策略表现优秀，参数稳健，建议实盘部署"
    elif grade_points >= 70:
        grade = "🟡 B级 - 观察池"
        recommendation = "策略基本合格，建议小仓位测试或继续优化"
    else:
        grade = "🔴 C级 - 淘汰"
        recommendation = "策略表现不佳，不建议实盘，需重新设计"

    print(f"\n最终评级: {grade}")
    print(f"建议: {recommendation}")

    return grade, grade_points

def main():
    # 测试参数
    symbol = 'TSLA'
    smoothing = 5  # 使用 fear_greed_index 表 (3因子指数)

    # 运行Walk-Forward分析
    wf_results, all_params, param_space, all_trades = walk_forward_analysis(symbol, smoothing)

    # 显示所有窗口结果
    print(f"\n{'='*80}")
    print("所有窗口详细结果")
    print(f"{'='*80}\n")
    print(wf_results.to_string(index=False))

    # 参数稳健性分析
    robust_zone, recommended = analyze_parameter_robustness(all_params)

    # 策略评级
    grade, points = evaluate_strategy(wf_results)

    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 保存窗口结果（移除trades列）
    wf_results_save = wf_results.drop(columns=['trades'], errors='ignore')
    filename_wf = f'../results/walk_forward_{symbol}_s{smoothing}_{timestamp}.csv'
    wf_results_save.to_csv(filename_wf, index=False)
    print(f"\n✅ Walk-Forward结果已保存: {filename_wf}")

    # 保存参数分布
    filename_params = f'../results/walk_forward_params_{symbol}_s{smoothing}_{timestamp}.csv'
    pd.DataFrame(all_params).to_csv(filename_params, index=False)
    print(f"✅ 参数分布已保存: {filename_params}")

    # 保存交易明细
    if all_trades:
        filename_trades = f'../results/walk_forward_trades_{symbol}_s{smoothing}_{timestamp}.csv'
        trades_df = pd.DataFrame(all_trades)
        trades_df.to_csv(filename_trades, index=False)
        print(f"✅ 交易明细已保存: {filename_trades}")

    print(f"\n{'='*80}")
    print("✅ Walk-Forward分析完成！")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
