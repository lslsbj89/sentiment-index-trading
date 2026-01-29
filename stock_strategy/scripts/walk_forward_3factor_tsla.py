#!/usr/bin/env python3
"""
Walk-Forward 参数优化：3因子恐惧贪婪指数策略 (TSLA)

3因子指数 = (PMACD + ROR + MoneyFlow) / 3, smoothing=5
数据存储在 fear_greed_index 表 (smoothing=5)

Walk-Forward 设计：
- 训练期：4年
- 测试期：1年
- 测试区间：2021-2025 (共5个窗口)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime
from itertools import product
from collections import Counter
from data_loader import DataLoader

db_config = {
    'host': 'localhost',
    'database': 'crypto_fear_greed_2',
    'user': 'sc2025',
    'password': '',
    'port': 5432
}


def load_3factor_index(symbol, start_date='2016-01-01', end_date='2025-12-31'):
    """加载3因子恐惧贪婪指数 (smoothing=5)"""
    conn = psycopg2.connect(**db_config)
    query = f"""
        SELECT date, smoothed_index
        FROM fear_greed_index
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


def backtest_strategy(prices, sentiment, buy_threshold, sell_and_threshold, sell_or_threshold,
                      use_stop_loss=True, stop_loss_pct=15, position_pct=0.8, initial_capital=100000):
    """
    回测策略

    买入条件：指数 < buy_threshold (恐惧时买入)
    卖出条件（优先级）：
    1. 止损：亏损达到 -stop_loss_pct%
    2. OR卖出：指数 > or_threshold (极度贪婪)
    3. AND卖出：指数 > and_threshold 且 价格 < MA50
    """
    df = prices.copy()
    df['idx'] = sentiment['smoothed_index']
    df['MA50'] = df['Close'].rolling(50).mean()
    df = df.dropna()

    if len(df) < 100:
        return None

    cash = initial_capital
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
            available = cash * position_pct
            shares = int(available / (current_price * 1.002))  # 0.2% 交易成本
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

            current_value = position * current_price
            entry_value = position * entry_price
            profit_pct = ((current_value - entry_value) / entry_value) * 100

            # 条件1: 止损保护
            if use_stop_loss and profit_pct <= -stop_loss_pct:
                sell_signal = True
                exit_reason = 'STOP_LOSS'
            # 条件2: OR卖出 (极度贪婪)
            elif current_idx > sell_or_threshold:
                sell_signal = True
                exit_reason = 'OR'
            # 条件3: AND卖出 (贪婪+价格弱势)
            elif current_idx > sell_and_threshold and current_price < current_ma50:
                sell_signal = True
                exit_reason = 'AND'

            if sell_signal:
                revenue = position * current_price * 0.998  # 0.2% 交易成本
                cash += revenue

                profit = revenue - (position * entry_price)
                profit_pct_final = (profit / (position * entry_price)) * 100

                trades.append({
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'entry_price': entry_price,
                    'exit_price': current_price * 0.998,
                    'profit': profit,
                    'profit_pct': profit_pct_final,
                    'exit_reason': exit_reason
                })

                position = 0

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
            'entry_price': entry_price,
            'exit_price': final_price * 0.998,
            'profit': profit,
            'profit_pct': profit_pct,
            'exit_reason': 'EOD'
        })
        position = 0

    total_value = cash
    portfolio_values.append(total_value)

    # 计算指标
    returns = pd.Series(portfolio_values).pct_change().dropna()
    total_return = ((total_value - initial_capital) / initial_capital) * 100
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 and returns.std() > 0 else 0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100
    max_dd = drawdown.min() if len(drawdown) > 0 else 0

    win_trades = [t for t in trades if t['profit_pct'] > 0]
    win_rate = len(win_trades) / len(trades) * 100 if len(trades) > 0 else 0

    # 计算盈亏比
    avg_win = np.mean([t['profit_pct'] for t in win_trades]) if len(win_trades) > 0 else 0
    lose_trades = [t for t in trades if t['profit_pct'] <= 0]
    avg_loss = abs(np.mean([t['profit_pct'] for t in lose_trades])) if len(lose_trades) > 0 else 1
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else avg_win

    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'num_trades': len(trades),
        'profit_loss_ratio': profit_loss_ratio,
        'trades': trades
    }


def grid_search_train(prices, sentiment, param_space, use_stop_loss=True):
    """网格搜索训练期最优参数"""
    results = []

    buy_vals = param_space['buy']
    and_vals = param_space['and']
    or_vals = param_space['or']

    total = len(buy_vals) * len(and_vals) * len(or_vals)
    count = 0

    for buy, and_val, or_val in product(buy_vals, and_vals, or_vals):
        count += 1
        if count % 20 == 0:
            print(f"    进度: {count}/{total}", end='\r')

        result = backtest_strategy(prices, sentiment, buy, and_val, or_val, use_stop_loss=use_stop_loss)

        if result is None:
            continue

        sharpe = result['sharpe']
        max_dd = result['max_dd']
        win_rate = result['win_rate'] / 100
        num_trades = result['num_trades']
        profit_loss_ratio = result['profit_loss_ratio']

        # 交易次数得分（鼓励适度交易）
        trade_score = min(num_trades / 10, 1.0)

        # 综合评分
        score = (0.35 * sharpe +
                 0.25 * (1 - abs(max_dd)/100) +
                 0.20 * win_rate +
                 0.10 * trade_score +
                 0.10 * min(profit_loss_ratio / 2, 1.0))

        results.append({
            'buy': buy,
            'and': and_val,
            'or': or_val,
            'sharpe': sharpe,
            'return': result['total_return'],
            'max_dd': max_dd,
            'win_rate': result['win_rate'],
            'num_trades': num_trades,
            'profit_loss_ratio': profit_loss_ratio,
            'score': score
        })

    print(f"    搜索完成: {len(results)} 个有效结果    ")

    if len(results) == 0:
        return None, None

    results_df = pd.DataFrame(results)
    best = results_df.loc[results_df['score'].idxmax()]

    return best, results_df


def walk_forward_analysis(symbol='TSLA', use_stop_loss=True, stop_loss_pct=15):
    """Walk-Forward分析"""

    print("=" * 80)
    print(f"Walk-Forward 参数优化：{symbol} (3因子恐惧贪婪指数, smoothing=5)")
    print("=" * 80)
    print(f"3因子 = (PMACD + ROR + MoneyFlow) / 3")
    print(f"止损保护: {'启用 (-' + str(stop_loss_pct) + '%)' if use_stop_loss else '关闭'}")
    print()

    # 加载数据
    print("加载数据...")
    loader = DataLoader(db_config)
    prices = loader.load_ohlcv(symbol, start_date='2016-01-01', end_date='2025-12-31')
    sentiment = load_3factor_index(symbol)
    loader.close()

    print(f"  价格数据: {prices.index.min().date()} ~ {prices.index.max().date()} ({len(prices)} 条)")
    print(f"  指数数据: {sentiment.index.min().date()} ~ {sentiment.index.max().date()} ({len(sentiment)} 条)")
    print(f"  指数范围: [{sentiment['smoothed_index'].min():.2f}, {sentiment['smoothed_index'].max():.2f}]")
    print(f"  指数均值: {sentiment['smoothed_index'].mean():.2f}")
    print()

    # 定义窗口：4年训练 + 1年测试
    windows = [
        {'name': 'Window1', 'train': ('2017-01-01', '2020-12-31'), 'test': ('2021-01-01', '2021-12-31')},
        {'name': 'Window2', 'train': ('2018-01-01', '2021-12-31'), 'test': ('2022-01-01', '2022-12-31')},
        {'name': 'Window3', 'train': ('2019-01-01', '2022-12-31'), 'test': ('2023-01-01', '2023-12-31')},
        {'name': 'Window4', 'train': ('2020-01-01', '2023-12-31'), 'test': ('2024-01-01', '2024-12-31')},
        {'name': 'Window5', 'train': ('2021-01-01', '2024-12-31'), 'test': ('2025-01-01', '2025-12-31')},
    ]

    # 参数搜索空间（根据3因子指数范围调整）
    # 指数范围：-44.57 ~ 108.82, 均值约10.76
    param_space = {
        'buy': [-20, -10, 0, 5, 10, 15, 20],      # 恐惧阈值（买入）
        'and': [20, 25, 30, 35, 40],              # AND卖出阈值
        'or': [50, 60, 70, 80, 90]                # OR卖出阈值
    }

    total_combos = len(param_space['buy']) * len(param_space['and']) * len(param_space['or'])
    print(f"参数搜索空间: {total_combos} 种组合")
    print(f"  buy阈值: {param_space['buy']}")
    print(f"  and阈值: {param_space['and']}")
    print(f"  or阈值: {param_space['or']}")
    print()

    # Walk-Forward循环
    wf_results = []
    all_train_params = []
    all_trades = []

    for window in windows:
        print("-" * 70)
        print(f"{window['name']}: 训练 {window['train'][0][:4]}-{window['train'][1][:4]} -> 测试 {window['test'][0][:4]}")
        print("-" * 70)

        # 分割数据
        train_start = pd.to_datetime(window['train'][0], utc=True)
        train_end = pd.to_datetime(window['train'][1], utc=True)
        test_start = pd.to_datetime(window['test'][0], utc=True)
        test_end = pd.to_datetime(window['test'][1], utc=True)

        train_prices = prices[(prices.index >= train_start) & (prices.index <= train_end)]
        train_sentiment = sentiment[(sentiment.index >= train_start) & (sentiment.index <= train_end)]

        test_prices = prices[(prices.index >= test_start) & (prices.index <= test_end)]
        test_sentiment = sentiment[(sentiment.index >= test_start) & (sentiment.index <= test_end)]

        print(f"  训练集: {len(train_prices)} 天, 测试集: {len(test_prices)} 天")

        # 训练期网格搜索
        print("  网格搜索中...")
        best_params, all_results = grid_search_train(train_prices, train_sentiment, param_space, use_stop_loss)

        if best_params is None:
            print("  ❌ 训练期无有效结果\n")
            continue

        print(f"  训练期最优参数: buy<{best_params['buy']}, and>{best_params['and']}, or>{best_params['or']}")
        print(f"  训练期Sharpe: {best_params['sharpe']:.3f}, 收益: {best_params['return']:.1f}%")

        all_train_params.append({
            'window': window['name'],
            'buy': best_params['buy'],
            'and': best_params['and'],
            'or': best_params['or'],
            'train_sharpe': best_params['sharpe'],
            'train_return': best_params['return']
        })

        # 测试期回测
        test_result = backtest_strategy(
            test_prices, test_sentiment,
            best_params['buy'], best_params['and'], best_params['or'],
            use_stop_loss=use_stop_loss, stop_loss_pct=stop_loss_pct
        )

        if test_result is None:
            print("  ❌ 测试期无有效结果\n")
            continue

        is_profitable = test_result['total_return'] > 0
        status = "盈利" if is_profitable else "亏损"

        print(f"  测试期结果: {test_result['total_return']:+.1f}% ({status}), "
              f"Sharpe: {test_result['sharpe']:.3f}, "
              f"回撤: {test_result['max_dd']:.1f}%, "
              f"交易: {test_result['num_trades']}次")
        print()

        # 记录交易明细
        for trade in test_result['trades']:
            trade['window'] = window['name']
            trade['test_year'] = window['test'][0][:4]
            all_trades.append(trade)

        wf_results.append({
            'window': window['name'],
            'train_period': f"{window['train'][0][:4]}-{window['train'][1][:4]}",
            'test_year': window['test'][0][:4],
            'buy': best_params['buy'],
            'and': best_params['and'],
            'or': best_params['or'],
            'train_sharpe': best_params['sharpe'],
            'train_return': best_params['return'],
            'test_return': test_result['total_return'],
            'test_sharpe': test_result['sharpe'],
            'test_max_dd': test_result['max_dd'],
            'test_win_rate': test_result['win_rate'],
            'test_trades': test_result['num_trades'],
            'profit_loss_ratio': test_result['profit_loss_ratio'],
            'is_profitable': is_profitable
        })

    # 汇总结果
    print("\n" + "=" * 80)
    print("Walk-Forward 测试结果汇总")
    print("=" * 80 + "\n")

    if len(wf_results) == 0:
        print("❌ 所有窗口均无有效结果")
        return None

    results_df = pd.DataFrame(wf_results)

    # 显示结果表格
    display_cols = ['window', 'test_year', 'buy', 'and', 'or',
                    'test_return', 'test_sharpe', 'test_max_dd', 'test_win_rate', 'test_trades']
    print(results_df[display_cols].to_string(index=False))

    # 参数稳健性分析
    print("\n" + "-" * 60)
    print("参数稳健性分析")
    print("-" * 60)

    buy_vals = [p['buy'] for p in all_train_params]
    and_vals = [p['and'] for p in all_train_params]
    or_vals = [p['or'] for p in all_train_params]

    print(f"\nbuy阈值: 均值={np.mean(buy_vals):.1f}, 标准差={np.std(buy_vals):.1f}, 分布={dict(Counter(buy_vals))}")
    print(f"and阈值: 均值={np.mean(and_vals):.1f}, 标准差={np.std(and_vals):.1f}, 分布={dict(Counter(and_vals))}")
    print(f"or阈值:  均值={np.mean(or_vals):.1f}, 标准差={np.std(or_vals):.1f}, 分布={dict(Counter(or_vals))}")

    # 性能统计
    print("\n" + "-" * 60)
    print("性能统计")
    print("-" * 60)

    profitable_windows = sum([r['is_profitable'] for r in wf_results])
    profitable_pct = profitable_windows / len(wf_results) * 100
    avg_return = np.mean([r['test_return'] for r in wf_results])
    avg_sharpe = np.mean([r['test_sharpe'] for r in wf_results])
    avg_dd = np.mean([r['test_max_dd'] for r in wf_results])
    avg_win_rate = np.mean([r['test_win_rate'] for r in wf_results])
    total_trades = sum([r['test_trades'] for r in wf_results])

    print(f"\n盈利窗口: {profitable_windows}/{len(wf_results)} ({profitable_pct:.0f}%)")
    print(f"平均年收益: {avg_return:.2f}%")
    print(f"平均Sharpe: {avg_sharpe:.3f}")
    print(f"平均最大回撤: {avg_dd:.1f}%")
    print(f"平均胜率: {avg_win_rate:.1f}%")
    print(f"总交易次数: {total_trades}")

    # 累计收益计算
    cumulative = 100000
    for r in wf_results:
        cumulative *= (1 + r['test_return'] / 100)
    cumulative_return = ((cumulative - 100000) / 100000) * 100

    print(f"\n累计收益 (2021-2025): {cumulative_return:.1f}%")
    print(f"最终资金: ${cumulative:,.0f}")

    # 策略评级
    print("\n" + "-" * 60)
    print("策略评级")
    print("-" * 60)

    score = 0

    # 盈利窗口率
    if profitable_pct >= 80:
        score += 40
        print(f"盈利窗口率 {profitable_pct:.0f}% (>=80%): +40分")
    elif profitable_pct >= 60:
        score += 25
        print(f"盈利窗口率 {profitable_pct:.0f}% (>=60%): +25分")
    elif profitable_pct >= 40:
        score += 10
        print(f"盈利窗口率 {profitable_pct:.0f}% (>=40%): +10分")
    else:
        print(f"盈利窗口率 {profitable_pct:.0f}% (<40%): +0分")

    # Sharpe
    if avg_sharpe >= 1.0:
        score += 30
        print(f"平均Sharpe {avg_sharpe:.2f} (>=1.0): +30分")
    elif avg_sharpe >= 0.5:
        score += 15
        print(f"平均Sharpe {avg_sharpe:.2f} (>=0.5): +15分")
    elif avg_sharpe >= 0:
        score += 5
        print(f"平均Sharpe {avg_sharpe:.2f} (>=0): +5分")
    else:
        print(f"平均Sharpe {avg_sharpe:.2f} (<0): +0分")

    # 回撤
    if avg_dd >= -20:
        score += 20
        print(f"平均回撤 {avg_dd:.1f}% (>=-20%): +20分")
    elif avg_dd >= -35:
        score += 10
        print(f"平均回撤 {avg_dd:.1f}% (>=-35%): +10分")
    else:
        print(f"平均回撤 {avg_dd:.1f}% (<-35%): +0分")

    # 收益
    if avg_return >= 30:
        score += 10
        print(f"平均收益 {avg_return:.1f}% (>=30%): +10分")
    elif avg_return >= 10:
        score += 5
        print(f"平均收益 {avg_return:.1f}% (>=10%): +5分")
    else:
        print(f"平均收益 {avg_return:.1f}% (<10%): +0分")

    print(f"\n总分: {score}/100")

    if score >= 80:
        rating = "A级 - 实盘推荐"
    elif score >= 60:
        rating = "B级 - 观察池"
    elif score >= 40:
        rating = "C级 - 需优化"
    else:
        rating = "D级 - 淘汰"

    print(f"评级: {rating}")

    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, f'wf_3factor_{symbol}_{timestamp}.csv')
    params_file = os.path.join(results_dir, f'wf_3factor_params_{symbol}_{timestamp}.csv')
    trades_file = os.path.join(results_dir, f'wf_3factor_trades_{symbol}_{timestamp}.csv')

    results_df.to_csv(results_file, index=False)
    pd.DataFrame(all_train_params).to_csv(params_file, index=False)
    if len(all_trades) > 0:
        pd.DataFrame(all_trades).to_csv(trades_file, index=False)

    print(f"\n结果已保存:")
    print(f"  {results_file}")
    print(f"  {params_file}")
    print(f"  {trades_file}")

    print("\n" + "=" * 80)
    print("Walk-Forward 分析完成!")
    print("=" * 80)

    return {
        'score': score,
        'rating': rating,
        'profitable_pct': profitable_pct,
        'avg_return': avg_return,
        'cumulative_return': cumulative_return,
        'avg_sharpe': avg_sharpe,
        'avg_dd': avg_dd,
        'results_df': results_df
    }


if __name__ == '__main__':
    # 运行TSLA的walk-forward分析
    result = walk_forward_analysis(symbol='TSLA', use_stop_loss=True, stop_loss_pct=15)
