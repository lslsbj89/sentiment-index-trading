#!/usr/bin/env python3
"""
最终对比: Smoothing=3 (优化后阈值) vs Smoothing=5

使用每只股票各自的最优阈值
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

# 每只股票的最优参数 (来自网格搜索)
OPTIMAL_PARAMS_S3 = {
    'NVDA': {'buy': 10, 'and': 30, 'or': 70},   # 收益率最优
    'TSLA': {'buy': -10, 'and': 25, 'or': 50},
    'GOOGL': {'buy': 10, 'and': 30, 'or': 70},
    'AAPL': {'buy': -10, 'and': 15, 'or': 40},
    'MSFT': {'buy': 0, 'and': 25, 'or': 40},
    'AMZN': {'buy': 0, 'and': 15, 'or': 60},
    'META': {'buy': 0, 'and': 15, 'or': 70}
}

# Smoothing=5 的统一参数
PARAMS_S5 = {'buy': 5, 'and': 20, 'or': 40}

def load_sentiment_index(symbol, smoothing=5):
    """加载情绪指数"""
    conn = psycopg2.connect(**db_config)
    table_name = "fear_greed_index_s3" if smoothing == 3 else "fear_greed_index"
    query = f"""
        SELECT date, smoothed_index
        FROM {table_name}
        WHERE symbol = '{symbol}'
          AND date >= '2020-01-01'
          AND date <= '2025-12-31'
        ORDER BY date
    """
    df = pd.read_sql(query, conn, parse_dates=['date'])
    conn.close()
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df.set_index('date', inplace=True)
    return df

def backtest_sentiment_strategy(prices, sentiment, buy_threshold, sell_and_threshold, sell_or_threshold):
    """情绪策略回测"""
    df = prices.copy()
    df['idx'] = sentiment['smoothed_index']
    df['MA50'] = df['Close'].rolling(50).mean()
    df = df.dropna()

    if len(df) < 100:
        return None

    cash = 100000
    position = 0
    portfolio_values = []

    for i in range(len(df)):
        current_price = df['Close'].iloc[i]
        current_idx = df['idx'].iloc[i]
        current_ma50 = df['MA50'].iloc[i]

        if position == 0 and current_idx < buy_threshold:
            available = cash * 0.8
            shares = int(available / (current_price * 1.002))
            if shares > 0:
                cost = shares * current_price * 1.002
                cash -= cost
                position = shares

        elif position > 0:
            sell_signal = False
            if current_idx > sell_or_threshold:
                sell_signal = True
            elif current_idx > sell_and_threshold and current_price < current_ma50:
                sell_signal = True

            if sell_signal:
                revenue = position * current_price * 0.998
                cash += revenue
                position = 0

        total_value = cash + position * current_price
        portfolio_values.append(total_value)

    if position > 0:
        final_price = df['Close'].iloc[-1]
        revenue = position * final_price * 0.998
        cash += revenue

    final_value = cash
    total_return = (final_value - 100000) / 100000 * 100

    portfolio_series = pd.Series(portfolio_values)
    cummax = portfolio_series.cummax()
    drawdown = (portfolio_series - cummax) / cummax * 100
    max_drawdown = drawdown.min()

    returns = portfolio_series.pct_change().dropna()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

    return {
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'final_value': final_value
    }

def main():
    print("\n" + "="*80)
    print("最终对比: Smoothing=3 (优化后) vs Smoothing=5")
    print("="*80)

    symbols = ['NVDA', 'TSLA', 'GOOGL', 'AAPL', 'MSFT', 'AMZN', 'META']

    results_s3_old = []  # S3 旧参数
    results_s3_opt = []  # S3 优化参数
    results_s5 = []      # S5

    for symbol in symbols:
        print(f"\n{'='*70}")
        print(f"测试 {symbol}")
        print(f"{'='*70}")

        try:
            loader = DataLoader(db_config)
            prices = loader.load_ohlcv(symbol, '2020-01-01', '2025-12-31')

            # S3 旧参数 (buy=5, and=20, or=40)
            sentiment_s3 = load_sentiment_index(symbol, smoothing=3)
            common_dates = prices.index.intersection(sentiment_s3.index)
            test_price = prices.loc[common_dates].loc['2021-01-01':]
            test_sentiment = sentiment_s3.loc[common_dates].loc['2021-01-01':]

            result_s3_old = backtest_sentiment_strategy(
                test_price, test_sentiment,
                buy_threshold=5, sell_and_threshold=20, sell_or_threshold=40
            )

            # S3 优化参数
            params = OPTIMAL_PARAMS_S3[symbol]
            result_s3_opt = backtest_sentiment_strategy(
                test_price, test_sentiment,
                buy_threshold=params['buy'],
                sell_and_threshold=params['and'],
                sell_or_threshold=params['or']
            )

            # S5
            sentiment_s5 = load_sentiment_index(symbol, smoothing=5)
            common_dates = prices.index.intersection(sentiment_s5.index)
            test_price = prices.loc[common_dates].loc['2021-01-01':]
            test_sentiment = sentiment_s5.loc[common_dates].loc['2021-01-01':]

            result_s5 = backtest_sentiment_strategy(
                test_price, test_sentiment,
                buy_threshold=PARAMS_S5['buy'],
                sell_and_threshold=PARAMS_S5['and'],
                sell_or_threshold=PARAMS_S5['or']
            )

            results_s3_old.append({'symbol': symbol, **result_s3_old})
            results_s3_opt.append({'symbol': symbol, **result_s3_opt})
            results_s5.append({'symbol': symbol, **result_s5})

            print(f"\nS3 (旧参数 buy<5, and>20, or>40):")
            print(f"  收益: {result_s3_old['total_return']:.2f}%")
            print(f"  夏普: {result_s3_old['sharpe_ratio']:.4f}")

            print(f"\nS3 (优化参数 buy<{params['buy']}, and>{params['and']}, or>{params['or']}):")
            print(f"  收益: {result_s3_opt['total_return']:.2f}%")
            print(f"  夏普: {result_s3_opt['sharpe_ratio']:.4f}")

            print(f"\nS5 (统一参数 buy<5, and>20, or>40):")
            print(f"  收益: {result_s5['total_return']:.2f}%")
            print(f"  夏普: {result_s5['sharpe_ratio']:.4f}")

            improve_old = result_s3_opt['total_return'] - result_s3_old['total_return']
            improve_s5 = result_s3_opt['total_return'] - result_s5['total_return']

            print(f"\n📊 改进:")
            print(f"  S3优化 vs S3旧: {improve_old:+.2f}%")
            print(f"  S3优化 vs S5: {improve_s5:+.2f}% ({'✅' if improve_s5 > 0 else '⚠️'})")

        except Exception as e:
            print(f"❌ {symbol} 失败: {e}")

    # 汇总
    print(f"\n{'='*80}")
    print("汇总对比")
    print(f"{'='*80}\n")

    df_s3_old = pd.DataFrame(results_s3_old)
    df_s3_opt = pd.DataFrame(results_s3_opt)
    df_s5 = pd.DataFrame(results_s5)

    comparison = pd.DataFrame({
        'symbol': df_s3_old['symbol'],
        'S3_old_return': df_s3_old['total_return'],
        'S3_opt_return': df_s3_opt['total_return'],
        'S5_return': df_s5['total_return'],
        'improve_old': df_s3_opt['total_return'] - df_s3_old['total_return'],
        'improve_s5': df_s3_opt['total_return'] - df_s5['total_return'],
        'S3_opt_sharpe': df_s3_opt['sharpe_ratio'],
        'S5_sharpe': df_s5['sharpe_ratio']
    })

    print(comparison.to_string(index=False))

    print(f"\n{'='*70}")
    print("统计摘要")
    print(f"{'='*70}")

    print(f"\n📈 平均收益率:")
    print(f"  S3 (旧参数): {comparison['S3_old_return'].mean():.2f}%")
    print(f"  S3 (优化后): {comparison['S3_opt_return'].mean():.2f}%")
    print(f"  S5: {comparison['S5_return'].mean():.2f}%")

    print(f"\n  优化改进: {comparison['improve_old'].mean():+.2f}%")
    diff_vs_s5 = comparison['improve_s5'].mean()
    winner = "✅ S3优化" if diff_vs_s5 > 0 else "⚠️ S5"
    print(f"  S3优化 vs S5: {diff_vs_s5:+.2f}% ({winner})")

    print(f"\n📊 平均夏普率:")
    print(f"  S3 (优化后): {comparison['S3_opt_sharpe'].mean():.4f}")
    print(f"  S5: {comparison['S5_sharpe'].mean():.4f}")

    print(f"\n🏆 胜负比 (S3优化 vs S5):")
    s3_wins = (comparison['improve_s5'] > 0).sum()
    s5_wins = (comparison['improve_s5'] < 0).sum()
    print(f"  S3 优化胜出: {s3_wins}/7 股票")
    print(f"  S5 胜出: {s5_wins}/7 股票")

    # 最终结论
    print(f"\n{'='*70}")
    print("🎯 最终结论")
    print(f"{'='*70}")

    if diff_vs_s5 > 10:
        print("\n✅ Smoothing=3 (优化后) 显著优于 Smoothing=5")
        print(f"   平均收益提升: {diff_vs_s5:.2f}%")
        print("\n   建议: 切换至 smoothing=3，但需要为每只股票使用各自的最优阈值")
    elif diff_vs_s5 > 0:
        print("\n⚖️ Smoothing=3 (优化后) 略优于 Smoothing=5")
        print(f"   平均收益提升: {diff_vs_s5:.2f}%")
        print("\n   建议: 可以考虑切换，但收益提升不大")
    else:
        print("\n⚠️ Smoothing=5 仍然更优")
        print(f"   即使优化阈值，S3 仍落后 {-diff_vs_s5:.2f}%")
        print("\n   建议: 维持 smoothing=5")

    # 保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    comparison.to_csv(f'final_comparison_optimized_s3_{timestamp}.csv', index=False)
    print(f"\n✅ 结果已保存: final_comparison_optimized_s3_{timestamp}.csv")

    print(f"\n{'='*80}")
    print("✅ 对比完成！")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
