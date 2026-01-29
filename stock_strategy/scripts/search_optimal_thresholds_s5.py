#!/usr/bin/env python3
"""
为 Smoothing=5 搜索每只股票的最优阈值

目标: 公平对比 S3 vs S5 (都使用个股优化参数)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

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

def grid_search_single_symbol(symbol, smoothing=5):
    print(f"\n{'='*70}")
    print(f"网格搜索: {symbol} (Smoothing={smoothing})")
    print(f"{'='*70}\n")

    print("加载数据...")
    loader = DataLoader(db_config)
    prices = loader.load_ohlcv(symbol, '2020-01-01', '2025-12-31')
    sentiment = load_sentiment_index(symbol, smoothing)

    common_dates = prices.index.intersection(sentiment.index)
    prices = prices.loc[common_dates]
    sentiment = sentiment.loc[common_dates]

    prices_test = prices.loc['2021-01-01':'2025-12-31']
    sentiment_test = sentiment.loc['2021-01-01':'2025-12-31']

    # Smoothing=5 的搜索范围（根据其分布调整）
    # Smoothing=5: min=-30, max=+50, std=14.6
    buy_thresholds = [-10, -5, 0, 5, 10]
    and_thresholds = [15, 20, 25, 30]
    or_thresholds = [35, 40, 45, 50]

    total_combinations = len(buy_thresholds) * len(and_thresholds) * len(or_thresholds)
    print(f"搜索空间: {total_combinations} 个参数组合")
    print(f"buy: {buy_thresholds}")
    print(f"and: {and_thresholds}")
    print(f"or:  {or_thresholds}\n")

    results = []
    count = 0

    for buy, and_t, or_t in product(buy_thresholds, and_thresholds, or_thresholds):
        count += 1
        if count % 10 == 0:
            print(f"进度: {count}/{total_combinations}", end='\r')

        try:
            result = backtest_sentiment_strategy(
                prices_test,
                sentiment_test,
                buy_threshold=buy,
                sell_and_threshold=and_t,
                sell_or_threshold=or_t
            )

            if result:
                results.append({
                    'symbol': symbol,
                    'buy': buy,
                    'and': and_t,
                    'or': or_t,
                    **result
                })
        except:
            pass

    print(f"\n完成: {len(results)} 个有效结果\n")

    df_results = pd.DataFrame(results)
    df_sorted_sharpe = df_results.sort_values('sharpe_ratio', ascending=False)
    df_sorted_return = df_results.sort_values('total_return', ascending=False)

    print("="*70)
    print(f"Top 5 参数组合 (按夏普率)")
    print("="*70)
    print(df_sorted_sharpe.head(5).to_string(index=False))

    print(f"\n{'='*70}")
    print(f"Top 5 参数组合 (按收益率)")
    print("="*70)
    print(df_sorted_return.head(5).to_string(index=False))

    best_sharpe = df_sorted_sharpe.iloc[0]
    best_return = df_sorted_return.iloc[0]

    print(f"\n{'='*70}")
    print("最优参数")
    print(f"{'='*70}")
    print(f"\n夏普率最优:")
    print(f"  buy < {best_sharpe['buy']}, and > {best_sharpe['and']}, or > {best_sharpe['or']}")
    print(f"  收益率: {best_sharpe['total_return']:.2f}%")
    print(f"  夏普率: {best_sharpe['sharpe_ratio']:.4f}")

    print(f"\n收益率最优:")
    print(f"  buy < {best_return['buy']}, and > {best_return['and']}, or > {best_return['or']}")
    print(f"  收益率: {best_return['total_return']:.2f}%")
    print(f"  夏普率: {best_return['sharpe_ratio']:.4f}")

    return df_results

def main():
    print("\n" + "="*80)
    print("Smoothing=5 个股优化参数搜索 - MAG7")
    print("="*80)

    symbols = ['NVDA', 'TSLA', 'GOOGL', 'AAPL', 'MSFT', 'AMZN', 'META']

    all_results = []

    for symbol in symbols:
        try:
            df_results = grid_search_single_symbol(symbol, smoothing=5)
            all_results.append(df_results)
        except Exception as e:
            print(f"❌ {symbol} 搜索失败: {e}")

    if all_results:
        print(f"\n{'='*80}")
        print("所有股票的最优参数汇总")
        print(f"{'='*80}\n")

        summary = []
        for i, symbol in enumerate(symbols[:len(all_results)]):
            df = all_results[i]
            best_sharpe = df.sort_values('sharpe_ratio', ascending=False).iloc[0]
            best_return = df.sort_values('total_return', ascending=False).iloc[0]

            summary.append({
                'symbol': symbol,
                'best_by_sharpe_buy': best_sharpe['buy'],
                'best_by_sharpe_and': best_sharpe['and'],
                'best_by_sharpe_or': best_sharpe['or'],
                'sharpe': best_sharpe['sharpe_ratio'],
                'return': best_sharpe['total_return'],
                'best_by_return_buy': best_return['buy'],
                'best_by_return_and': best_return['and'],
                'best_by_return_or': best_return['or'],
                'return2': best_return['total_return'],
                'sharpe2': best_return['sharpe_ratio']
            })

        df_summary = pd.DataFrame(summary)
        print(df_summary.to_string(index=False))

        print(f"\n{'='*70}")
        print("参数频率分析")
        print(f"{'='*70}")

        print(f"\nbuy 阈值分布 (按夏普):")
        print(df_summary['best_by_sharpe_buy'].value_counts().sort_index())

        print(f"\nand 阈值分布 (按夏普):")
        print(df_summary['best_by_sharpe_and'].value_counts().sort_index())

        print(f"\nor 阈值分布 (按夏普):")
        print(df_summary['best_by_sharpe_or'].value_counts().sort_index())

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        for i, symbol in enumerate(symbols[:len(all_results)]):
            all_results[i].to_csv(
                f'grid_search_s5_{symbol}_{timestamp}.csv',
                index=False
            )

        df_summary.to_csv(f'grid_search_s5_summary_{timestamp}.csv', index=False)

        print(f"\n✅ 结果已保存:")
        print(f"  - grid_search_s5_summary_{timestamp}.csv (汇总)")
        print(f"  - grid_search_s5_{{SYMBOL}}_{timestamp}.csv (各股详情)")

    print(f"\n{'='*80}")
    print("✅ 搜索完成！")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
