"""
实验D: cycle_index 六只股票分别优化
为 NVDA, AAPL, GOOGL, META, MSFT, AMZN 分别找最优参数
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
from backtest_engine import EnhancedBacktester

# 配置
SYMBOLS = ['NVDA', 'AAPL', 'GOOGL', 'META', 'MSFT', 'AMZN']
db_config = {
    "host": "localhost",
    "port": 5432,
    "database": "crypto_fear_greed_2",
    "user": "sc2025",
    "password": ""
}

# 网格搜索参数
BUY_THRESHOLDS = [-25, -20, -15, -10, -5, 0]
AND_SELL_THRESHOLDS = [0, 5, 10, 15, 20]
OR_SELL_THRESHOLDS = [20, 25, 30, 35, 40]

# 测试年份
TEST_YEARS = [2021, 2022, 2023, 2024, 2025]

# 回测参数
BACKTEST_PARAMS = {
    "initial_capital": 100000,
    "commission_rate": 0.001,
    "slippage_rate": 0.001,
    "take_profit_pct": 999.0,
    "stop_loss_pct": 999.0,
    "max_holding_days": 999,
    "position_pct": 0.8
}


def load_cycle_index(db_config, symbol):
    conn = psycopg2.connect(**db_config)
    query = f"""
        SELECT date, smoothed_index
        FROM cycle_index
        WHERE symbol = '{symbol}'
        ORDER BY date
    """
    df = pd.read_sql(query, conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.set_index('date')
    return df


def load_price_with_ma(db_config, symbol):
    loader = DataLoader(db_config)
    ohlcv = loader.load_ohlcv(symbol, start_date="2015-01-01")
    loader.close()
    ohlcv['MA50'] = ohlcv['Close'].rolling(window=50).mean()
    return ohlcv


def run_single_year_backtest(price_data, sentiment_data, buy_th, and_sell_th, or_sell_th, year):
    start_date = pd.Timestamp(f"{year}-01-01", tz='UTC')
    end_date = pd.Timestamp(f"{year}-12-31", tz='UTC')

    mask = (price_data.index >= start_date) & (price_data.index <= end_date)
    test_price = price_data[mask].copy()
    test_sentiment = sentiment_data.reindex(test_price.index)

    if len(test_price) == 0:
        return 0

    signals = pd.DataFrame(index=test_price.index)
    signals['smoothed_index'] = test_sentiment['smoothed_index']
    signals['Close'] = test_price['Close']
    signals['MA50'] = test_price['MA50']

    signals['buy_signal'] = (signals['smoothed_index'] < buy_th).astype(int)
    and_condition = (signals['smoothed_index'] > and_sell_th) & (signals['Close'] < signals['MA50'])
    or_condition = signals['smoothed_index'] > or_sell_th
    signals['sell_signal'] = (and_condition | or_condition).astype(int)
    signals['prob_profit'] = 0.5
    signals['position_size'] = 0

    backtester = EnhancedBacktester(
        initial_capital=BACKTEST_PARAMS["initial_capital"],
        commission_rate=BACKTEST_PARAMS["commission_rate"],
        slippage_rate=BACKTEST_PARAMS["slippage_rate"],
        take_profit_pct=BACKTEST_PARAMS["take_profit_pct"],
        stop_loss_pct=BACKTEST_PARAMS["stop_loss_pct"],
        max_holding_days=BACKTEST_PARAMS["max_holding_days"],
        use_dynamic_position=True,
        position_pct=BACKTEST_PARAMS["position_pct"]
    )

    portfolio, metrics, trades = backtester.run_backtest_with_sell_signal(signals, test_price)
    if metrics:
        return metrics.get('total_return', 0) * 100
    return 0


def optimize_single_stock(symbol, sentiment_data, price_data):
    """为单只股票做网格搜索"""
    results = []

    for buy_th, and_sell_th, or_sell_th in product(BUY_THRESHOLDS, AND_SELL_THRESHOLDS, OR_SELL_THRESHOLDS):
        yearly_returns = {}
        for year in TEST_YEARS:
            ret = run_single_year_backtest(price_data, sentiment_data, buy_th, and_sell_th, or_sell_th, year)
            yearly_returns[year] = ret

        returns_list = list(yearly_returns.values())
        avg_return = np.mean(returns_list)
        cumulative = np.prod([1 + r/100 for r in returns_list]) * 100 - 100
        positive_years = sum(1 for r in returns_list if r > 0)

        result = {
            'buy_threshold': buy_th,
            'and_sell_threshold': and_sell_th,
            'or_sell_threshold': or_sell_th,
            **{f'y{year}': yearly_returns[year] for year in TEST_YEARS},
            'avg_return': avg_return,
            'cumulative': cumulative,
            'positive_years': positive_years
        }
        results.append(result)

    return pd.DataFrame(results)


def main():
    print("=" * 100)
    print("实验D: cycle_index 六只股票分别优化")
    print("=" * 100)
    print(f"\n搜索范围:")
    print(f"  买入阈值:      {BUY_THRESHOLDS}")
    print(f"  卖出阈值1(AND): {AND_SELL_THRESHOLDS}")
    print(f"  卖出阈值2(OR):  {OR_SELL_THRESHOLDS}")

    total_combinations = len(BUY_THRESHOLDS) * len(AND_SELL_THRESHOLDS) * len(OR_SELL_THRESHOLDS)
    print(f"  每只股票组合数: {total_combinations}")

    # 存储所有股票的最优结果
    all_best_results = []

    for symbol in SYMBOLS:
        print(f"\n{'='*100}")
        print(f"优化 {symbol}...")
        print(f"{'='*100}")

        # 加载数据
        sentiment_data = load_cycle_index(db_config, symbol)
        price_data = load_price_with_ma(db_config, symbol)

        # 网格搜索
        results_df = optimize_single_stock(symbol, sentiment_data, price_data)
        results_df = results_df.sort_values('cumulative', ascending=False)

        # 最优参数
        best = results_df.iloc[0]

        print(f"\n【{symbol} 最优参数】")
        print(f"  买入: 情绪 < {best['buy_threshold']}")
        print(f"  卖出: (情绪 > {best['and_sell_threshold']} AND 价格 < MA50) OR (情绪 > {best['or_sell_threshold']})")
        print(f"  ─────────────────────────────────────────")
        print(f"  2021: {best['y2021']:+.1f}%")
        print(f"  2022: {best['y2022']:+.1f}%")
        print(f"  2023: {best['y2023']:+.1f}%")
        print(f"  2024: {best['y2024']:+.1f}%")
        print(f"  2025: {best['y2025']:+.1f}%")
        print(f"  ─────────────────────────────────────────")
        print(f"  平均: {best['avg_return']:+.1f}%/年")
        print(f"  5年累计: {best['cumulative']:+.1f}%")
        print(f"  盈利年: {int(best['positive_years'])}/5")

        # 保存该股票的最优结果
        best_result = {
            'symbol': symbol,
            'buy_threshold': best['buy_threshold'],
            'and_sell_threshold': best['and_sell_threshold'],
            'or_sell_threshold': best['or_sell_threshold'],
            'y2021': best['y2021'],
            'y2022': best['y2022'],
            'y2023': best['y2023'],
            'y2024': best['y2024'],
            'y2025': best['y2025'],
            'avg_return': best['avg_return'],
            'cumulative': best['cumulative'],
            'positive_years': best['positive_years']
        }
        all_best_results.append(best_result)

        # 保存该股票的完整搜索结果
        stock_file = f"cycle_index_optimize_{symbol}.csv"
        results_df.to_csv(stock_file, index=False)

    # 添加 TSLA 的最优结果
    tsla_best = {
        'symbol': 'TSLA',
        'buy_threshold': -5,
        'and_sell_threshold': 15,
        'or_sell_threshold': 30,
        'y2021': 91.4,
        'y2022': -41.3,
        'y2023': 133.3,
        'y2024': 76.9,
        'y2025': 56.3,
        'avg_return': 63.3,
        'cumulative': 624.8,
        'positive_years': 4
    }
    all_best_results.append(tsla_best)

    # 汇总表
    print("\n" + "=" * 100)
    print("七姐妹最优参数汇总 (按5年累计排序)")
    print("=" * 100)

    summary_df = pd.DataFrame(all_best_results)
    summary_df = summary_df.sort_values('cumulative', ascending=False)

    print(f"\n{'股票':<6} {'买入':<6} {'AND':<5} {'OR':<5} {'2021':>7} {'2022':>7} {'2023':>7} {'2024':>7} {'2025':>7} {'平均':>7} {'累计':>8} {'盈利':>5}")
    print("-" * 95)
    for _, row in summary_df.iterrows():
        print(f"{row['symbol']:<6} <{row['buy_threshold']:<4} >{row['and_sell_threshold']:<4} >{row['or_sell_threshold']:<4} "
              f"{row['y2021']:>+6.0f}% {row['y2022']:>+6.0f}% {row['y2023']:>+6.0f}% {row['y2024']:>+6.0f}% {row['y2025']:>+6.0f}% "
              f"{row['avg_return']:>+6.0f}% {row['cumulative']:>+7.0f}% {int(row['positive_years']):>4}/5")

    # 计算总平均
    print("-" * 95)
    avg_cumulative = summary_df['cumulative'].mean()
    avg_return = summary_df['avg_return'].mean()
    print(f"{'平均':<6} {'':14} {summary_df['y2021'].mean():>+6.0f}% {summary_df['y2022'].mean():>+6.0f}% "
          f"{summary_df['y2023'].mean():>+6.0f}% {summary_df['y2024'].mean():>+6.0f}% {summary_df['y2025'].mean():>+6.0f}% "
          f"{avg_return:>+6.0f}% {avg_cumulative:>+7.0f}%")

    # 与旧指数对比
    print("\n" + "=" * 100)
    print("新指数 vs 旧指数 对比")
    print("=" * 100)

    old_results = {
        'NVDA': 602, 'TSLA': 329, 'GOOGL': 199,
        'AAPL': 125, 'META': 109, 'MSFT': 94, 'AMZN': 52
    }

    print(f"\n{'股票':<6} {'旧指数':>10} {'新指数':>10} {'差异':>10}")
    print("-" * 40)
    total_old = 0
    total_new = 0
    for _, row in summary_df.iterrows():
        symbol = row['symbol']
        old_val = old_results.get(symbol, 0)
        new_val = row['cumulative']
        diff = new_val - old_val
        total_old += old_val
        total_new += new_val
        better = "✅" if diff > 0 else "❌" if diff < -50 else ""
        print(f"{symbol:<6} {old_val:>+9.0f}% {new_val:>+9.0f}% {diff:>+9.0f}% {better}")

    print("-" * 40)
    print(f"{'平均':<6} {total_old/7:>+9.0f}% {total_new/7:>+9.0f}% {(total_new-total_old)/7:>+9.0f}%")

    # 保存汇总结果
    output_file = f"cycle_index_mag7_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    summary_df.to_csv(output_file, index=False)
    print(f"\n汇总结果已保存: {output_file}")


if __name__ == "__main__":
    main()
