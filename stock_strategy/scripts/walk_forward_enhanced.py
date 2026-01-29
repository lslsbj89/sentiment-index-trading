#!/usr/bin/env python3
"""
Walk-Forward 情绪指数策略验证（增强版）

新增功能：
1. 市场过滤：SPY跌破MA200时空仓
2. 止损保护：单笔亏损达到-15%时强制止损
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

def load_sentiment_index(symbol, smoothing=3):
    """加载情绪指数"""
    conn = psycopg2.connect(**db_config)
    table_name = "fear_greed_index_s3" if smoothing == 3 else "fear_greed_index"
    query = f"""
        SELECT date, smoothed_index
        FROM {table_name}
        WHERE symbol = '{symbol}'
          AND date >= '2016-01-01'
          AND date <= '2025-12-31'
        ORDER BY date
    """
    df = pd.read_sql(query, conn, parse_dates=['date'])
    conn.close()
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df.set_index('date', inplace=True)
    return df

def backtest_strategy_enhanced(prices, sentiment, spy_prices,
                               buy_threshold, sell_and_threshold, sell_or_threshold,
                               use_market_filter=False, use_stop_loss=False, stop_loss_pct=15):
    """
    回测策略（增强版）

    新增参数：
    - use_market_filter: 是否使用SPY MA200市场过滤
    - use_stop_loss: 是否使用止损保护
    - stop_loss_pct: 止损百分比（默认15%）

    卖出条件：
    1. OR卖出：指数 > or_threshold
    2. AND卖出：指数 > and_threshold 且 价格 < MA50
    3. 止损卖出：亏损达到 -stop_loss_pct% (可选)
    4. 期末平仓：回测结束仍有持仓
    """
    df = prices.copy()
    df['idx'] = sentiment['smoothed_index']
    df['MA50'] = df['Close'].rolling(50).mean()

    # 加入SPY MA200（市场过滤）
    if use_market_filter:
        df['SPY_Close'] = spy_prices['Close']
        df['SPY_MA200'] = df['SPY_Close'].rolling(200).mean()

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

        # 市场过滤：检查SPY是否在MA200之上
        market_ok = True
        if use_market_filter:
            spy_price = df['SPY_Close'].iloc[i]
            spy_ma200 = df['SPY_MA200'].iloc[i]
            market_ok = spy_price > spy_ma200

        # 买入逻辑（加入市场过滤）
        if position == 0 and current_idx < buy_threshold and market_ok:
            available = cash * 0.8
            shares = int(available / (current_price * 1.002))
            if shares > 0:
                cost = shares * current_price * 1.002
                cash -= cost
                position = shares
                entry_price = current_price * 1.002
                entry_date = current_date

        # 卖出逻辑
        elif position > 0:
            sell_signal = False
            exit_reason = None

            # 计算当前盈亏
            current_value = position * current_price
            entry_value = position * entry_price
            profit_pct = ((current_value - entry_value) / entry_value) * 100

            # 条件1: 止损保护（优先级最高）
            if use_stop_loss and profit_pct <= -stop_loss_pct:
                sell_signal = True
                exit_reason = 'STOP_LOSS'
            # 条件2: OR卖出
            elif current_idx > sell_or_threshold:
                sell_signal = True
                exit_reason = 'OR'
            # 条件3: AND卖出
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
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'exit_reason': exit_reason
                })

                position = 0

        # 记录组合价值
        total_value = cash + position * current_price
        portfolio_values.append(total_value)

    # 期末平仓
    if position > 0:
        final_price = df['Close'].iloc[-1]
        revenue = position * final_price * 0.998
        cash += revenue

        profit = revenue - (position * entry_price)
        profit_pct = (profit / (position * entry_price)) * 100

        trades.append({
            'entry_date': entry_date,
            'exit_date': df.index[-1],
            'profit': profit,
            'profit_pct': profit_pct,
            'exit_reason': 'EOD'
        })
        position = 0

    total_value = cash + position * df['Close'].iloc[-1]
    portfolio_values.append(total_value)

    # 计算指标
    returns = pd.Series(portfolio_values).pct_change().dropna()
    total_return = ((total_value - 100000) / 100000) * 100
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 and returns.std() > 0 else 0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100
    max_dd = drawdown.min()

    win_trades = [t for t in trades if t['profit_pct'] > 0]
    win_rate = len(win_trades) / len(trades) * 100 if len(trades) > 0 else 0

    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'num_trades': len(trades),
        'trades': trades
    }

def grid_search_train(prices, sentiment, spy_prices, param_space,
                     use_market_filter=False, use_stop_loss=False):
    """网格搜索训练期最优参数"""
    results = []

    buy_vals = param_space['buy']
    and_vals = param_space['and']
    or_vals = param_space['or']

    total = len(buy_vals) * len(and_vals) * len(or_vals)
    count = 0

    for buy, and_val, or_val in product(buy_vals, and_vals, or_vals):
        count += 1
        if count % 10 == 0:
            print(f"  搜索进度: {count}/{total}", end='  ')

        result = backtest_strategy_enhanced(prices, sentiment, spy_prices, buy, and_val, or_val,
                                           use_market_filter, use_stop_loss)

        if result is None:
            continue

        # 自定义评分函数
        sharpe = result['sharpe']
        max_dd = result['max_dd']
        win_rate = result['win_rate'] / 100
        num_trades = result['num_trades']

        trade_score = min(num_trades / 10, 1.0)

        score = (0.4 * sharpe +
                0.3 * (1 - abs(max_dd)/100) +
                0.2 * win_rate +
                0.1 * trade_score)

        results.append({
            'buy': buy,
            'and': and_val,
            'or': or_val,
            'sharpe': sharpe,
            'return': result['total_return'],
            'max_dd': max_dd,
            'win_rate': win_rate,
            'num_trades': num_trades,
            'score': score
        })

    print(f"  搜索完成: {len(results)} 个有效结果\n")

    if len(results) == 0:
        return None

    results_df = pd.DataFrame(results)
    best = results_df.loc[results_df['score'].idxmax()]

    return best, results_df

def walk_forward_analysis(symbol, smoothing=3, use_market_filter=False, use_stop_loss=False):
    """Walk-Forward分析"""

    version_name = "基础版"
    if use_market_filter and use_stop_loss:
        version_name = "完全增强版"
    elif use_market_filter:
        version_name = "市场过滤版"
    elif use_stop_loss:
        version_name = "止损保护版"

    print("="*80)
    print(f"Walk-Forward 分析: {symbol} (Smoothing={smoothing}) - {version_name}")
    print("="*80)
    if use_market_filter:
        print("✅ 市场过滤: SPY < MA200 时不买入")
    if use_stop_loss:
        print("✅ 止损保护: 单笔亏损 ≥ -15% 时强制止损")
    print()

    # 加载数据
    print("加载数据...")
    loader = DataLoader(db_config)
    prices = loader.load_ohlcv(symbol, start_date='2016-01-01', end_date='2025-12-31')
    spy_prices = loader.load_ohlcv('SPY', start_date='2016-01-01', end_date='2025-12-31')
    sentiment = load_sentiment_index(symbol, smoothing)
    loader.close()
    print("✓ 成功连接到数据库: crypto_fear_greed_2\n")

    # 定义窗口
    windows = [
        {'name': 'Window1', 'train': ('2017-01-01', '2020-12-31'), 'test': ('2021-01-01', '2021-12-31')},
        {'name': 'Window2', 'train': ('2018-01-01', '2021-12-31'), 'test': ('2022-01-01', '2022-12-31')},
        {'name': 'Window3', 'train': ('2019-01-01', '2022-12-31'), 'test': ('2023-01-01', '2023-12-31')},
        {'name': 'Window4', 'train': ('2020-01-01', '2023-12-31'), 'test': ('2024-01-01', '2024-12-31')},
        {'name': 'Window5', 'train': ('2021-01-01', '2024-12-31'), 'test': ('2025-01-01', '2025-12-31')},
    ]

    # 参数搜索空间（修正后的120组合）
    param_space = {
        'buy': [-15, -10, -5, 0, 5, 10],
        'and': [10, 15, 20, 25, 30],
        'or': [40, 50, 60, 70]
    }

    # Walk-Forward循环
    wf_results = []
    all_train_params = []

    for window in windows:
        print(f"\n{'='*70}")
        print(f"{window['name']}: 训练{window['train'][0][:4]}-{window['train'][1][:4]} → 测试{window['test'][0][:4]}")
        print("="*70 + "\n")

        # 分割数据
        train_start = pd.to_datetime(window['train'][0], utc=True)
        train_end = pd.to_datetime(window['train'][1], utc=True)
        test_start = pd.to_datetime(window['test'][0], utc=True)
        test_end = pd.to_datetime(window['test'][1], utc=True)

        train_prices = prices[(prices.index >= train_start) & (prices.index <= train_end)]
        train_sentiment = sentiment[(sentiment.index >= train_start) & (sentiment.index <= train_end)]
        train_spy = spy_prices[(spy_prices.index >= train_start) & (spy_prices.index <= train_end)]

        test_prices = prices[(prices.index >= test_start) & (prices.index <= test_end)]
        test_sentiment = sentiment[(sentiment.index >= test_start) & (sentiment.index <= test_end)]
        test_spy = spy_prices[(spy_prices.index >= test_start) & (spy_prices.index <= test_end)]

        # 训练期网格搜索
        print("📊 训练期网格搜索...")
        best_params, all_results = grid_search_train(train_prices, train_sentiment, train_spy,
                                                     param_space, use_market_filter, use_stop_loss)

        if best_params is None:
            print("❌ 训练期无有效结果\n")
            continue

        print(f"✅ 训练期最优参数:")
        print(f"   buy < {best_params['buy']}, and > {best_params['and']}, or > {best_params['or']}")
        print(f"   训练期Sharpe: {best_params['sharpe']:.4f}")
        print(f"   训练期收益: {best_params['return']:.2f}%")
        print(f"   评分: {best_params['score']:.4f}\n")

        all_train_params.append({
            'window': window['name'],
            'buy': best_params['buy'],
            'and': best_params['and'],
            'or': best_params['or']
        })

        # 测试期回测
        print("📈 测试期回测（用训练期最优参数）...\n")
        test_result = backtest_strategy_enhanced(
            test_prices, test_sentiment, test_spy,
            best_params['buy'], best_params['and'], best_params['or'],
            use_market_filter, use_stop_loss
        )

        if test_result is None:
            print("❌ 测试期无有效结果\n")
            continue

        is_profitable = test_result['total_return'] > 0
        status = "✅ 盈利" if is_profitable else "❌ 亏损"

        print(f"✅ 测试期结果:")
        print(f"   收益: {test_result['total_return']:.2f}%")
        print(f"   Sharpe: {test_result['sharpe']:.4f}")
        print(f"   最大回撤: {test_result['max_dd']:.2f}%")
        print(f"   胜率: {test_result['win_rate']:.1f}%")
        print(f"   交易次数: {test_result['num_trades']}")
        print(f"   状态: {status}\n")

        wf_results.append({
            'window': window['name'],
            'train_period': f"{window['train'][0][:4]}-{window['train'][1][:4]}",
            'test_period': window['test'][0][:4],
            'train_sharpe': best_params['sharpe'],
            'train_return': best_params['return'],
            'test_return': test_result['total_return'],
            'test_sharpe': test_result['sharpe'],
            'test_max_dd': test_result['max_dd'],
            'test_win_rate': test_result['win_rate'],
            'test_trades': test_result['num_trades'],
            'is_profitable': is_profitable,
            'buy': best_params['buy'],
            'and': best_params['and'],
            'or': best_params['or']
        })

    # 汇总结果
    print("\n" + "="*80)
    print("所有窗口详细结果")
    print("="*80 + "\n")

    results_df = pd.DataFrame(wf_results)
    print(results_df.to_string(index=False))

    # 参数稳健性分析
    print("\n" + "="*80)
    print("参数稳健性分析")
    print("="*80 + "\n")

    print("📊 各参数统计:\n")

    buy_vals = [p['buy'] for p in all_train_params]
    and_vals = [p['and'] for p in all_train_params]
    or_vals = [p['or'] for p in all_train_params]

    print(f"buy阈值:")
    print(f"  均值: {np.mean(buy_vals):.1f}")
    print(f"  标准差: {np.std(buy_vals):.1f}")
    print(f"  范围: [{min(buy_vals)}, {max(buy_vals)}]")
    from collections import Counter
    print(f"  分布: {dict(Counter(buy_vals))}\n")

    print(f"and阈值:")
    print(f"  均值: {np.mean(and_vals):.1f}")
    print(f"  标准差: {np.std(and_vals):.1f}")
    print(f"  范围: [{min(and_vals)}, {max(and_vals)}]")
    print(f"  分布: {dict(Counter(and_vals))}\n")

    print(f"or阈值:")
    print(f"  均值: {np.mean(or_vals):.1f}")
    print(f"  标准差: {np.std(or_vals):.1f}")
    print(f"  范围: [{min(or_vals)}, {max(or_vals)}]")
    print(f"  分布: {dict(Counter(or_vals))}\n")

    # 策略评级
    print("\n" + "="*80)
    print("策略最终评级")
    print("="*80 + "\n")

    if len(wf_results) == 0:
        print("⚠️ 警告: 所有测试窗口均无有效结果")
        print("建议: 市场过滤条件过于严格，导致无交易机会\n")
        return {
            'score': 0,
            'rating': '🔴 C级 - 淘汰',
            'profitable_pct': 0,
            'avg_return': 0,
            'avg_sharpe': 0,
            'avg_dd': 0
        }

    profitable_pct = sum([r['is_profitable'] for r in wf_results]) / len(wf_results) * 100
    avg_return = np.mean([r['test_return'] for r in wf_results])
    avg_sharpe = np.mean([r['test_sharpe'] for r in wf_results])
    avg_dd = np.mean([r['test_max_dd'] for r in wf_results])
    avg_win_rate = np.mean([r['test_win_rate'] for r in wf_results])

    print(f"📊 Walk-Forward 总体表现:")
    print(f"  窗口数: {len(wf_results)}")
    print(f"  盈利窗口率: {profitable_pct:.1f}% ({sum([r['is_profitable'] for r in wf_results])}/{len(wf_results)})")
    print(f"  平均测试收益: {avg_return:.2f}%")
    print(f"  平均测试Sharpe: {avg_sharpe:.4f}")
    print(f"  平均最大回撤: {avg_dd:.2f}%")
    print(f"  平均胜率: {avg_win_rate:.1f}%\n")

    # 评分
    score = 0
    print(f"🎯 策略评级:")

    if profitable_pct >= 80:
        score += 40
        print(f"  ✅ 盈利窗口率 {profitable_pct:.1f}% (≥80%): +40分")
    elif profitable_pct >= 60:
        score += 20
        print(f"  🟡 盈利窗口率 {profitable_pct:.1f}% (≥60%): +20分")
    else:
        print(f"  ❌ 盈利窗口率 {profitable_pct:.1f}% (<60%): +0分")

    if avg_sharpe >= 1.0:
        score += 30
        print(f"  ✅ 平均Sharpe {avg_sharpe:.2f} (≥1.0): +30分")
    elif avg_sharpe >= 0.5:
        score += 15
        print(f"  🟡 平均Sharpe {avg_sharpe:.2f} (≥0.5): +15分")
    else:
        print(f"  ❌ 平均Sharpe {avg_sharpe:.2f} (<0.5): +0分")

    if avg_dd >= -25:
        score += 20
        print(f"  ✅ 平均回撤 {avg_dd:.1f}% (≥-25%): +20分")
    elif avg_dd >= -35:
        score += 10
        print(f"  🟡 平均回撤 {avg_dd:.1f}% (≥-35%): +10分")
    else:
        print(f"  ❌ 平均回撤 {avg_dd:.1f}% (<-35%): +0分")

    if avg_return >= 20:
        score += 10
        print(f"  ✅ 平均收益 {avg_return:.1f}% (≥20%): +10分")
    elif avg_return >= 0:
        score += 5
        print(f"  🟡 平均收益 {avg_return:.1f}% (≥0%): +5分")
    else:
        print(f"  ❌ 平均收益 {avg_return:.1f}% (<0%): +0分")

    print(f"\n总分: {score}/100\n")

    if score >= 85:
        rating = "🟢 A级 - 实盘推荐"
        suggestion = "策略表现优秀，建议实盘部署"
    elif score >= 70:
        rating = "🟡 B级 - 观察池"
        suggestion = "策略基本合格，建议小仓位测试或继续优化"
    else:
        rating = "🔴 C级 - 淘汰"
        suggestion = "策略表现不佳，建议重新设计或放弃"

    print(f"最终评级: {rating}")
    print(f"建议: {suggestion}\n")

    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = ""
    if use_market_filter and use_stop_loss:
        suffix = "_enhanced"
    elif use_market_filter:
        suffix = "_market_filter"
    elif use_stop_loss:
        suffix = "_stop_loss"

    results_file = f'../results/walk_forward_{symbol}_s{smoothing}{suffix}_{timestamp}.csv'
    params_file = f'../results/walk_forward_params_{symbol}_s{smoothing}{suffix}_{timestamp}.csv'

    results_df.to_csv(results_file, index=False)
    pd.DataFrame(all_train_params).to_csv(params_file, index=False)

    print(f"✅ Walk-Forward结果已保存: {results_file}")
    print(f"✅ 参数分布已保存: {params_file}\n")

    print("="*80)
    print("✅ Walk-Forward分析完成！")
    print("="*80)

    return {
        'score': score,
        'rating': rating,
        'profitable_pct': profitable_pct,
        'avg_return': avg_return,
        'avg_sharpe': avg_sharpe,
        'avg_dd': avg_dd
    }

if __name__ == '__main__':
    # 测试TSLA，Smoothing=3
    symbol = 'TSLA'
    smoothing = 3

    print("\n" + "="*80)
    print("🔬 对比实验：基础版 vs 增强版")
    print("="*80 + "\n")

    # 1. 基础版（当前版本）
    print("【实验1】基础版（无增强功能）\n")
    result_baseline = walk_forward_analysis(symbol, smoothing,
                                           use_market_filter=False,
                                           use_stop_loss=False)

    print("\n" + "="*80)
    print("继续下一个实验...")
    print("="*80 + "\n")

    # 2. 市场过滤版
    print("【实验2】市场过滤版（SPY MA200过滤）\n")
    result_market = walk_forward_analysis(symbol, smoothing,
                                         use_market_filter=True,
                                         use_stop_loss=False)

    print("\n" + "="*80)
    print("继续下一个实验...")
    print("="*80 + "\n")

    # 3. 止损保护版
    print("【实验3】止损保护版（-15%止损）\n")
    result_stoploss = walk_forward_analysis(symbol, smoothing,
                                           use_market_filter=False,
                                           use_stop_loss=True)

    print("\n" + "="*80)
    print("继续下一个实验...")
    print("="*80 + "\n")

    # 4. 完全增强版
    print("【实验4】完全增强版（市场过滤 + 止损保护）\n")
    result_full = walk_forward_analysis(symbol, smoothing,
                                       use_market_filter=True,
                                       use_stop_loss=True)

    # 最终对比
    print("\n" + "="*80)
    print("📊 最终对比总结")
    print("="*80 + "\n")

    comparison = pd.DataFrame([
        {
            '版本': '基础版',
            '评分': result_baseline['score'],
            '评级': result_baseline['rating'],
            '盈利率': f"{result_baseline['profitable_pct']:.1f}%",
            '平均收益': f"{result_baseline['avg_return']:.2f}%",
            'Sharpe': f"{result_baseline['avg_sharpe']:.4f}",
            '平均回撤': f"{result_baseline['avg_dd']:.2f}%"
        },
        {
            '版本': '市场过滤版',
            '评分': result_market['score'],
            '评级': result_market['rating'],
            '盈利率': f"{result_market['profitable_pct']:.1f}%",
            '平均收益': f"{result_market['avg_return']:.2f}%",
            'Sharpe': f"{result_market['avg_sharpe']:.4f}",
            '平均回撤': f"{result_market['avg_dd']:.2f}%"
        },
        {
            '版本': '止损保护版',
            '评分': result_stoploss['score'],
            '评级': result_stoploss['rating'],
            '盈利率': f"{result_stoploss['profitable_pct']:.1f}%",
            '平均收益': f"{result_stoploss['avg_return']:.2f}%",
            'Sharpe': f"{result_stoploss['avg_sharpe']:.4f}",
            '平均回撤': f"{result_stoploss['avg_dd']:.2f}%"
        },
        {
            '版本': '完全增强版',
            '评分': result_full['score'],
            '评级': result_full['rating'],
            '盈利率': f"{result_full['profitable_pct']:.1f}%",
            '平均收益': f"{result_full['avg_return']:.2f}%",
            'Sharpe': f"{result_full['avg_sharpe']:.4f}",
            '平均回撤': f"{result_full['avg_dd']:.2f}%"
        }
    ])

    print(comparison.to_string(index=False))
    print("\n" + "="*80)
    print("✅ 全部对比实验完成！")
    print("="*80)
