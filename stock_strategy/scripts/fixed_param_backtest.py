#!/usr/bin/env python3
"""
固定参数回测脚本 - 用于验证跨时期泛化能力

用途: 使用2017-2025推荐的参数，回测2016-2020期间
目标: 验证参数是否真正跨时期稳健
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import pandas as pd
import numpy as np
import psycopg2
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
          AND date >= '2012-01-01'
          AND date <= '2020-12-31'
        ORDER BY date
    """
    df = pd.read_sql(query, conn, parse_dates=['date'])
    conn.close()
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df.set_index('date', inplace=True)
    return df

def backtest_strategy(prices, sentiment, buy_threshold, sell_and_threshold, sell_or_threshold):
    """
    回测策略（固定参数）

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

        # 卖出逻辑
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

def evaluate_result(result):
    """评估策略表现并打分"""
    if result is None:
        return 0, 'F'

    score = 0

    # 盈利能力（40分）
    if result['total_return'] > 0:
        profitable = True
        if result['total_return'] >= 100:
            score += 40
        elif result['total_return'] >= 50:
            score += 30
        elif result['total_return'] >= 20:
            score += 20
        else:
            score += 10
    else:
        profitable = False

    # Sharpe比率（30分）
    if result['sharpe_ratio'] >= 1.0:
        score += 30
    elif result['sharpe_ratio'] >= 0.8:
        score += 20
    elif result['sharpe_ratio'] >= 0.5:
        score += 10

    # 最大回撤（20分）
    if abs(result['max_drawdown']) <= 15:
        score += 20
    elif abs(result['max_drawdown']) <= 25:
        score += 10

    # 胜率（10分）
    if result['win_rate'] >= 0.7:
        score += 10
    elif result['win_rate'] >= 0.5:
        score += 5

    # 评级
    if score >= 85:
        grade = 'A'
    elif score >= 70:
        grade = 'B'
    else:
        grade = 'C'

    return score, grade, profitable

def main():
    """
    测试2017-2025推荐参数在2016-2020的表现
    """
    # 2017-2025推荐参数
    test_params = {
        'MSFT': {'buy': 0, 'and': 15, 'or': 50},
        'AAPL': {'buy': -2, 'and': 10, 'or': 40},
        'NVDA': {'buy': 5, 'and': 15, 'or': 50},
        'TSLA': {'buy': -5, 'and': 15, 'or': 50},
    }

    print("="*80)
    print("反向测试: 2017-2025推荐参数 → 2016-2020期间回测")
    print("="*80)
    print("\n目的: 验证参数跨时期泛化能力\n")

    results = []

    for symbol, params in test_params.items():
        print(f"\n{'='*70}")
        print(f"测试 {symbol}")
        print(f"{'='*70}")
        print(f"参数: buy<{params['buy']}, and>{params['and']}, or>{params['or']}")
        print(f"来源: 2017-2025推荐参数")
        print(f"测试期: 2016-2020")

        # 加载数据
        print("\n加载数据...")
        loader = DataLoader(db_config)
        prices = loader.load_ohlcv(symbol, '2012-01-01', '2020-12-31')
        sentiment = load_sentiment_index(symbol, smoothing=3)

        # 对齐数据
        common_dates = prices.index.intersection(sentiment.index)
        prices = prices.loc[common_dates]
        sentiment = sentiment.loc[common_dates]

        # 筛选2016-2020期间
        prices_test = prices.loc['2016-01-01':'2020-12-31']
        sentiment_test = sentiment.loc['2016-01-01':'2020-12-31']

        print(f"测试期数据点: {len(prices_test)}")

        # 回测
        print("\n执行回测...")
        result = backtest_strategy(
            prices_test,
            sentiment_test,
            params['buy'],
            params['and'],
            params['or']
        )

        if result:
            score, grade, profitable = evaluate_result(result)

            print(f"\n{'='*70}")
            print("回测结果:")
            print(f"{'='*70}")
            print(f"总收益: {result['total_return']:.2f}%")
            print(f"Sharpe比率: {result['sharpe_ratio']:.4f}")
            print(f"最大回撤: {result['max_drawdown']:.2f}%")
            print(f"胜率: {result['win_rate']*100:.1f}%")
            print(f"交易次数: {result['num_trades']}")
            print(f"是否盈利: {'✅ 是' if profitable else '❌ 否'}")
            print(f"\n评分: {score}/100")
            print(f"评级: {grade}级")

            results.append({
                'symbol': symbol,
                'buy': params['buy'],
                'and': params['and'],
                'or': params['or'],
                'return': result['total_return'],
                'sharpe': result['sharpe_ratio'],
                'max_dd': result['max_drawdown'],
                'win_rate': result['win_rate'],
                'trades': result['num_trades'],
                'profitable': profitable,
                'score': score,
                'grade': grade
            })
        else:
            print("❌ 数据不足，无法回测")

    # 总结
    print(f"\n{'='*80}")
    print("测试总结")
    print(f"{'='*80}\n")

    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

    # 保存结果
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'../results/reverse_test_2017params_on_2016-2020_{timestamp}.csv'
    df_results.to_csv(filename, index=False)
    print(f"\n✅ 结果已保存: {filename}")

    return df_results

if __name__ == "__main__":
    main()
