#!/usr/bin/env python3
"""
公平对比: Smoothing=3 vs Smoothing=5

对比两种场景：
1. 个股优化 vs 个股优化 (公平)
2. 统一参数 vs 统一参数 (公平)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime
from data_loader import DataLoader

db_config = {
    'host': 'localhost',
    'database': 'crypto_fear_greed_2',
    'user': 'sc2025',
    'password': '',
    'port': 5432
}

# 每只股票的最优参数 (来自网格搜索)
OPTIMAL_PARAMS_S3 = {
    'NVDA': {'buy': 10, 'and': 30, 'or': 70},
    'TSLA': {'buy': -10, 'and': 25, 'or': 50},
    'GOOGL': {'buy': 10, 'and': 30, 'or': 70},
    'AAPL': {'buy': -10, 'and': 15, 'or': 40},
    'MSFT': {'buy': 0, 'and': 25, 'or': 40},
    'AMZN': {'buy': 0, 'and': 15, 'or': 60},
    'META': {'buy': 0, 'and': 15, 'or': 70}
}

OPTIMAL_PARAMS_S5 = {
    'NVDA': {'buy': 10, 'and': 30, 'or': 50},
    'TSLA': {'buy': -10, 'and': 20, 'or': 35},
    'GOOGL': {'buy': 10, 'and': 30, 'or': 50},
    'AAPL': {'buy': -5, 'and': 25, 'or': 35},
    'MSFT': {'buy': 10, 'and': 30, 'or': 50},
    'AMZN': {'buy': 5, 'and': 15, 'or': 35},
    'META': {'buy': -10, 'and': 15, 'or': 45}
}

# 统一参数
UNIFIED_S3 = {'buy': 0, 'and': 20, 'or': 60}  # S3的统一折中参数
UNIFIED_S5 = {'buy': 5, 'and': 20, 'or': 40}  # S5的统一最优参数

def load_sentiment_index(symbol, smoothing=5):
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
    print("公平对比: Smoothing=3 vs Smoothing=5")
    print("="*80)

    symbols = ['NVDA', 'TSLA', 'GOOGL', 'AAPL', 'MSFT', 'AMZN', 'META']

    results_s3_opt = []  # S3 个股优化
    results_s5_opt = []  # S5 个股优化
    results_s3_uni = []  # S3 统一参数
    results_s5_uni = []  # S5 统一参数

    for symbol in symbols:
        print(f"\n{'='*70}")
        print(f"测试 {symbol}")
        print(f"{'='*70}")

        try:
            loader = DataLoader(db_config)
            prices = loader.load_ohlcv(symbol, '2020-01-01', '2025-12-31')

            # S3 个股优化
            sentiment_s3 = load_sentiment_index(symbol, smoothing=3)
            common_dates = prices.index.intersection(sentiment_s3.index)
            test_price = prices.loc[common_dates].loc['2021-01-01':]
            test_sentiment = sentiment_s3.loc[common_dates].loc['2021-01-01':]

            params = OPTIMAL_PARAMS_S3[symbol]
            result_s3_opt = backtest_sentiment_strategy(
                test_price, test_sentiment,
                params['buy'], params['and'], params['or']
            )

            # S3 统一参数
            result_s3_uni = backtest_sentiment_strategy(
                test_price, test_sentiment,
                UNIFIED_S3['buy'], UNIFIED_S3['and'], UNIFIED_S3['or']
            )

            # S5 个股优化
            sentiment_s5 = load_sentiment_index(symbol, smoothing=5)
            common_dates = prices.index.intersection(sentiment_s5.index)
            test_price = prices.loc[common_dates].loc['2021-01-01':]
            test_sentiment = sentiment_s5.loc[common_dates].loc['2021-01-01':]

            params = OPTIMAL_PARAMS_S5[symbol]
            result_s5_opt = backtest_sentiment_strategy(
                test_price, test_sentiment,
                params['buy'], params['and'], params['or']
            )

            # S5 统一参数
            result_s5_uni = backtest_sentiment_strategy(
                test_price, test_sentiment,
                UNIFIED_S5['buy'], UNIFIED_S5['and'], UNIFIED_S5['or']
            )

            results_s3_opt.append({'symbol': symbol, **result_s3_opt})
            results_s5_opt.append({'symbol': symbol, **result_s5_opt})
            results_s3_uni.append({'symbol': symbol, **result_s3_uni})
            results_s5_uni.append({'symbol': symbol, **result_s5_uni})

            print(f"\n  个股优化:")
            print(f"    S3: {result_s3_opt['total_return']:.2f}% (夏普 {result_s3_opt['sharpe_ratio']:.4f})")
            print(f"    S5: {result_s5_opt['total_return']:.2f}% (夏普 {result_s5_opt['sharpe_ratio']:.4f})")
            diff_opt = result_s3_opt['total_return'] - result_s5_opt['total_return']
            winner = "✅ S3" if diff_opt > 0 else "⚠️ S5"
            print(f"    差异: {diff_opt:+.2f}% ({winner})")

            print(f"\n  统一参数:")
            print(f"    S3: {result_s3_uni['total_return']:.2f}% (夏普 {result_s3_uni['sharpe_ratio']:.4f})")
            print(f"    S5: {result_s5_uni['total_return']:.2f}% (夏普 {result_s5_uni['sharpe_ratio']:.4f})")
            diff_uni = result_s3_uni['total_return'] - result_s5_uni['total_return']
            winner = "✅ S3" if diff_uni > 0 else "⚠️ S5"
            print(f"    差异: {diff_uni:+.2f}% ({winner})")

        except Exception as e:
            print(f"❌ {symbol} 失败: {e}")

    # 汇总对比
    print(f"\n{'='*80}")
    print("场景1: 个股优化 vs 个股优化 (公平对比)")
    print(f"{'='*80}\n")

    df_s3_opt = pd.DataFrame(results_s3_opt)
    df_s5_opt = pd.DataFrame(results_s5_opt)

    comparison_opt = pd.DataFrame({
        'symbol': df_s3_opt['symbol'],
        'S3_opt_return': df_s3_opt['total_return'],
        'S5_opt_return': df_s5_opt['total_return'],
        'diff': df_s3_opt['total_return'] - df_s5_opt['total_return'],
        'S3_opt_sharpe': df_s3_opt['sharpe_ratio'],
        'S5_opt_sharpe': df_s5_opt['sharpe_ratio']
    })

    print(comparison_opt.to_string(index=False))

    print(f"\n{'='*70}")
    print("统计摘要 (个股优化)")
    print(f"{'='*70}")

    print(f"\n📈 平均收益率:")
    print(f"  S3 (个股优化): {comparison_opt['S3_opt_return'].mean():.2f}%")
    print(f"  S5 (个股优化): {comparison_opt['S5_opt_return'].mean():.2f}%")
    diff_opt = comparison_opt['diff'].mean()
    winner = "✅ S3" if diff_opt > 0 else "⚠️ S5"
    print(f"  差异: {diff_opt:+.2f}% ({winner} 更优)")

    print(f"\n📊 平均夏普率:")
    print(f"  S3: {comparison_opt['S3_opt_sharpe'].mean():.4f}")
    print(f"  S5: {comparison_opt['S5_opt_sharpe'].mean():.4f}")

    s3_wins = (comparison_opt['diff'] > 0).sum()
    s5_wins = (comparison_opt['diff'] < 0).sum()
    print(f"\n🏆 胜负比:")
    print(f"  S3 胜出: {s3_wins}/7 股票")
    print(f"  S5 胜出: {s5_wins}/7 股票")

    # 统一参数对比
    print(f"\n{'='*80}")
    print("场景2: 统一参数 vs 统一参数 (公平对比)")
    print(f"{'='*80}\n")

    df_s3_uni = pd.DataFrame(results_s3_uni)
    df_s5_uni = pd.DataFrame(results_s5_uni)

    comparison_uni = pd.DataFrame({
        'symbol': df_s3_uni['symbol'],
        'S3_uni_return': df_s3_uni['total_return'],
        'S5_uni_return': df_s5_uni['total_return'],
        'diff': df_s3_uni['total_return'] - df_s5_uni['total_return'],
        'S3_uni_sharpe': df_s3_uni['sharpe_ratio'],
        'S5_uni_sharpe': df_s5_uni['sharpe_ratio']
    })

    print(comparison_uni.to_string(index=False))

    print(f"\n{'='*70}")
    print("统计摘要 (统一参数)")
    print(f"{'='*70}")

    print(f"\n📈 平均收益率:")
    print(f"  S3 (统一 buy<0, and>20, or>60): {comparison_uni['S3_uni_return'].mean():.2f}%")
    print(f"  S5 (统一 buy<5, and>20, or>40): {comparison_uni['S5_uni_return'].mean():.2f}%")
    diff_uni = comparison_uni['diff'].mean()
    winner = "✅ S3" if diff_uni > 0 else "⚠️ S5"
    print(f"  差异: {diff_uni:+.2f}% ({winner} 更优)")

    print(f"\n📊 平均夏普率:")
    print(f"  S3: {comparison_uni['S3_uni_sharpe'].mean():.4f}")
    print(f"  S5: {comparison_uni['S5_uni_sharpe'].mean():.4f}")

    s3_wins = (comparison_uni['diff'] > 0).sum()
    s5_wins = (comparison_uni['diff'] < 0).sum()
    print(f"\n🏆 胜负比:")
    print(f"  S3 胜出: {s3_wins}/7 股票")
    print(f"  S5 胜出: {s5_wins}/7 股票")

    # 最终结论
    print(f"\n{'='*80}")
    print("🎯 最终结论")
    print(f"{'='*80}")

    print(f"\n场景1 (个股优化 vs 个股优化):")
    if diff_opt > 10:
        print(f"  ✅ S3 显著优于 S5 (+{diff_opt:.2f}%)")
    elif diff_opt > 0:
        print(f"  ⚖️ S3 略优于 S5 (+{diff_opt:.2f}%)")
    else:
        print(f"  ⚠️ S5 优于 S3 ({diff_opt:.2f}%)")

    print(f"\n场景2 (统一参数 vs 统一参数):")
    if diff_uni > 10:
        print(f"  ✅ S3 显著优于 S5 (+{diff_uni:.2f}%)")
    elif diff_uni > 0:
        print(f"  ⚖️ S3 略优于 S5 (+{diff_uni:.2f}%)")
    else:
        print(f"  ⚠️ S5 优于 S3 ({diff_uni:.2f}%)")

    print(f"\n综合建议:")
    if diff_opt > 0 and diff_uni > 0:
        print("  ✅ 建议切换至 Smoothing=3")
        if diff_opt > diff_uni + 20:
            print("  ✅ 优先使用个股优化参数 (收益提升更大)")
        else:
            print("  ✅ 可使用统一参数 (管理更简便)")
    elif diff_opt > 0:
        print("  ⚖️ 如需个股优化，可选择 S3；如需统一参数，选择 S5")
    else:
        print("  ⚠️ 建议维持 Smoothing=5")

    # 保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    comparison_opt.to_csv(f'fair_comparison_optimized_{timestamp}.csv', index=False)
    comparison_uni.to_csv(f'fair_comparison_unified_{timestamp}.csv', index=False)

    print(f"\n✅ 结果已保存:")
    print(f"  - fair_comparison_optimized_{timestamp}.csv (个股优化对比)")
    print(f"  - fair_comparison_unified_{timestamp}.csv (统一参数对比)")

    print(f"\n{'='*80}")
    print("✅ 公平对比完成！")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
