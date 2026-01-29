#!/usr/bin/env python3
"""
Smoothing=3 vs Smoothing=5 对比实验

利用数据库中已有的数据:
- fear_greed_index_s3 (smoothing=3)
- fear_greed_index (smoothing=5)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime
from data_loader import DataLoader

# 数据库配置
db_config = {
    'host': 'localhost',
    'database': 'crypto_fear_greed_2',
    'user': 'sc2025',
    'password': '',
    'port': 5432
}

def load_sentiment_index(symbol, smoothing=5, start_date='2020-01-01', end_date='2025-12-31'):
    """加载情绪指数"""
    conn = psycopg2.connect(**db_config)

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

def backtest_sentiment_strategy(
    prices,
    sentiment,
    buy_threshold=5,
    sell_and_threshold=20,
    sell_or_threshold=40,
    position_pct=0.8,
    initial_capital=100000
):
    """情绪策略回测"""

    # 合并数据
    df = prices.copy()
    df['idx'] = sentiment['smoothed_index']
    df['MA50'] = df['Close'].rolling(50).mean()
    df = df.dropna()

    if len(df) < 100:
        return None

    # 初始化
    cash = initial_capital
    position = 0
    entry_price = 0
    trades = []
    portfolio_values = []

    for i in range(len(df)):
        current_price = df['Close'].iloc[i]
        current_idx = df['idx'].iloc[i]
        current_ma50 = df['MA50'].iloc[i]
        current_date = df.index[i]

        # 买入条件
        if position == 0 and current_idx < buy_threshold:
            available = cash * position_pct
            shares = int(available / (current_price * 1.002))  # 0.2% 成本
            if shares > 0:
                cost = shares * current_price * 1.002
                cash -= cost
                position = shares
                entry_price = current_price
                entry_date = current_date
                entry_idx = current_idx

        # 卖出条件
        elif position > 0:
            sell_signal = False
            exit_reason = ''

            if current_idx > sell_or_threshold:
                sell_signal = True
                exit_reason = f'idx>{sell_or_threshold}'
            elif current_idx > sell_and_threshold and current_price < current_ma50:
                sell_signal = True
                exit_reason = f'idx>{sell_and_threshold} & <MA50'

            if sell_signal:
                revenue = position * current_price * 0.998  # 0.2% 成本
                profit_pct = (current_price - entry_price) / entry_price * 100
                holding_days = (current_date - entry_date).days

                trades.append({
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'entry_idx': entry_idx,
                    'exit_date': current_date,
                    'exit_price': current_price,
                    'exit_idx': current_idx,
                    'profit_pct': profit_pct,
                    'holding_days': holding_days,
                    'exit_reason': exit_reason
                })

                cash += revenue
                position = 0

        # 记录组合价值
        portfolio_value = cash + position * current_price
        portfolio_values.append(portfolio_value)

    # 期末强制平仓
    if position > 0:
        final_price = df['Close'].iloc[-1]
        revenue = position * final_price * 0.998
        cash += revenue

    # 计算指标
    final_value = cash
    total_return = (final_value - initial_capital) / initial_capital * 100

    # 最大回撤
    portfolio_series = pd.Series(portfolio_values)
    cummax = portfolio_series.cummax()
    drawdown = (portfolio_series - cummax) / cummax * 100
    max_drawdown = drawdown.min()

    # 夏普率
    returns = portfolio_series.pct_change().dropna()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

    # 交易统计
    trades_df = pd.DataFrame(trades)
    num_trades = len(trades_df)
    win_rate = (trades_df['profit_pct'] > 0).sum() / num_trades * 100 if num_trades > 0 else 0

    return {
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'final_value': final_value,
        'trades': trades_df
    }

def test_single_symbol(symbol, smoothing):
    """测试单个股票"""

    print(f"\n  Smoothing={smoothing}:")

    try:
        # 加载数据
        loader = DataLoader(db_config)
        prices = loader.load_ohlcv(symbol, '2020-01-01', '2025-12-31')
        sentiment = load_sentiment_index(symbol, smoothing, '2020-01-01', '2025-12-31')

        # 对齐数据
        common_dates = prices.index.intersection(sentiment.index)
        prices = prices.loc[common_dates]
        sentiment = sentiment.loc[common_dates]

        # 回测 2021-2025
        prices_test = prices.loc['2021-01-01':'2025-12-31']
        sentiment_test = sentiment.loc['2021-01-01':'2025-12-31']

        result = backtest_sentiment_strategy(
            prices_test,
            sentiment_test,
            buy_threshold=5,
            sell_and_threshold=20,
            sell_or_threshold=40,
            position_pct=0.8,
            initial_capital=100000
        )

        if result:
            print(f"    收益率: {result['total_return']:.2f}%")
            print(f"    夏普率: {result['sharpe_ratio']:.4f}")
            print(f"    最大回撤: {result['max_drawdown']:.2f}%")
            print(f"    交易次数: {result['num_trades']}")
            print(f"    胜率: {result['win_rate']:.2f}%")
            print(f"    最终资金: ${result['final_value']:.2f}")

            return result
        else:
            print("    ❌ 数据不足")
            return None

    except Exception as e:
        print(f"    ❌ 失败: {e}")
        return None

def main():
    """主测试函数"""

    print("\n" + "="*80)
    print("Smoothing=3 vs Smoothing=5 对比实验")
    print("测试周期: 2021-2025")
    print("="*80)

    symbols = ['NVDA', 'TSLA', 'GOOGL', 'AAPL', 'MSFT', 'AMZN', 'META']

    results_s3 = []
    results_s5 = []

    for symbol in symbols:
        print(f"\n{'='*70}")
        print(f"测试 {symbol}")
        print(f"{'='*70}")

        # 测试 smoothing=3
        result_s3 = test_single_symbol(symbol, smoothing=3)
        if result_s3:
            results_s3.append({
                'symbol': symbol,
                **{k: v for k, v in result_s3.items() if k != 'trades'}
            })

        # 测试 smoothing=5
        result_s5 = test_single_symbol(symbol, smoothing=5)
        if result_s5:
            results_s5.append({
                'symbol': symbol,
                **{k: v for k, v in result_s5.items() if k != 'trades'}
            })

        # 对比
        if result_s3 and result_s5:
            return_diff = result_s3['total_return'] - result_s5['total_return']
            sharpe_diff = result_s3['sharpe_ratio'] - result_s5['sharpe_ratio']

            print(f"\n  📊 对比:")
            winner_icon = "✅ s3" if return_diff > 0 else "⚠️ s5"
            print(f"    收益差异: {return_diff:+.2f}% ({winner_icon}更优)")

            winner_icon = "✅ s3" if sharpe_diff > 0 else "⚠️ s5"
            print(f"    夏普差异: {sharpe_diff:+.4f} ({winner_icon}更优)")

            trade_diff = result_s3['num_trades'] - result_s5['num_trades']
            print(f"    交易次数差异: {trade_diff:+d}")

    # 汇总
    if results_s3 and results_s5:
        print(f"\n{'='*80}")
        print("汇总对比分析")
        print(f"{'='*80}\n")

        df_s3 = pd.DataFrame(results_s3)
        df_s5 = pd.DataFrame(results_s5)

        comparison = pd.DataFrame({
            'symbol': df_s3['symbol'],
            'return_s3': df_s3['total_return'],
            'return_s5': df_s5['total_return'],
            'return_diff': df_s3['total_return'] - df_s5['total_return'],
            'sharpe_s3': df_s3['sharpe_ratio'],
            'sharpe_s5': df_s5['sharpe_ratio'],
            'sharpe_diff': df_s3['sharpe_ratio'] - df_s5['sharpe_ratio'],
            'trades_s3': df_s3['num_trades'],
            'trades_s5': df_s5['num_trades'],
            'win_rate_s3': df_s3['win_rate'],
            'win_rate_s5': df_s5['win_rate']
        })

        print(comparison.to_string(index=False))

        # 统计
        print(f"\n{'='*70}")
        print("统计摘要")
        print(f"{'='*70}")

        print(f"\n📈 平均收益率:")
        print(f"  Smoothing=3: {comparison['return_s3'].mean():.2f}%")
        print(f"  Smoothing=5: {comparison['return_s5'].mean():.2f}%")
        diff = comparison['return_diff'].mean()
        winner = "✅ Smoothing=3" if diff > 0 else "⚠️ Smoothing=5"
        print(f"  差异: {diff:+.2f}% ({winner} 更优)")

        print(f"\n📊 平均夏普率:")
        print(f"  Smoothing=3: {comparison['sharpe_s3'].mean():.4f}")
        print(f"  Smoothing=5: {comparison['sharpe_s5'].mean():.4f}")
        diff = comparison['sharpe_diff'].mean()
        winner = "✅ Smoothing=3" if diff > 0 else "⚠️ Smoothing=5"
        print(f"  差异: {diff:+.4f} ({winner} 更优)")

        print(f"\n🔄 平均交易次数:")
        print(f"  Smoothing=3: {comparison['trades_s3'].mean():.1f}")
        print(f"  Smoothing=5: {comparison['trades_s5'].mean():.1f}")
        print(f"  差异: {comparison['trades_s3'].mean() - comparison['trades_s5'].mean():+.1f}")

        print(f"\n🎯 平均胜率:")
        print(f"  Smoothing=3: {comparison['win_rate_s3'].mean():.2f}%")
        print(f"  Smoothing=5: {comparison['win_rate_s5'].mean():.2f}%")

        # 胜负统计
        s3_wins = (comparison['return_diff'] > 0).sum()
        s5_wins = (comparison['return_diff'] < 0).sum()

        print(f"\n🏆 胜负比 (按收益):")
        print(f"  Smoothing=3 胜出: {s3_wins}/7 股票")
        print(f"  Smoothing=5 胜出: {s5_wins}/7 股票")

        # 最终结论
        print(f"\n{'='*70}")
        print("🎯 结论")
        print(f"{'='*70}")

        if comparison['return_diff'].mean() > 5:
            print("\n✅ Smoothing=3 明显更优 (平均收益提升 >5%)")
            print("   建议: 全面切换至 smoothing=3")
        elif comparison['return_diff'].mean() > 0:
            print("\n⚖️ Smoothing=3 略优 (平均收益提升 <5%)")
            print("   建议: 可以考虑切换，或分股票使用")
        else:
            print("\n⚠️ Smoothing=5 更优")
            print("   建议: 维持 smoothing=5")

        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        df_s3.to_csv(f'mag7_smoothing3_{timestamp}.csv', index=False)
        df_s5.to_csv(f'mag7_smoothing5_{timestamp}.csv', index=False)
        comparison.to_csv(f'smoothing_comparison_mag7_{timestamp}.csv', index=False)

        print(f"\n✅ 结果已保存:")
        print(f"  - mag7_smoothing3_{timestamp}.csv")
        print(f"  - mag7_smoothing5_{timestamp}.csv")
        print(f"  - smoothing_comparison_mag7_{timestamp}.csv")

    print(f"\n{'='*80}")
    print("✅ 实验完成！")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
