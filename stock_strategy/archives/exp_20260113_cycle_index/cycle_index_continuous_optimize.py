"""
实验F: cycle_index 连续测试专用优化
针对连续5年测试场景，为各股票找最优参数
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
SYMBOLS = ['NVDA', 'TSLA', 'AAPL', 'GOOGL', 'META', 'MSFT', 'AMZN']
db_config = {
    "host": "localhost",
    "port": 5432,
    "database": "crypto_fear_greed_2",
    "user": "sc2025",
    "password": ""
}

# 网格搜索参数 (针对连续测试调整)
BUY_THRESHOLDS = [-15, -10, -5, 0, 5]
AND_SELL_THRESHOLDS = [5, 10, 15, 20]
OR_SELL_THRESHOLDS = [20, 25, 30, 35, 40]

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


def run_continuous_backtest(price_data, sentiment_data, buy_th, and_sell_th, or_sell_th):
    """运行连续5年回测"""
    start_date = pd.Timestamp("2021-01-01", tz='UTC')
    end_date = pd.Timestamp("2025-12-31", tz='UTC')

    mask = (price_data.index >= start_date) & (price_data.index <= end_date)
    test_price = price_data[mask].copy()
    test_sentiment = sentiment_data.reindex(test_price.index)

    if len(test_price) == 0:
        return 0, 0, 0

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
        return (
            metrics.get('total_return', 0) * 100,
            metrics.get('max_drawdown', 0) * 100,
            metrics.get('total_trades', 0)
        )
    return 0, 0, 0


def optimize_single_stock(symbol, sentiment_data, price_data):
    """为单只股票做连续测试优化"""
    results = []

    for buy_th, and_sell_th, or_sell_th in product(BUY_THRESHOLDS, AND_SELL_THRESHOLDS, OR_SELL_THRESHOLDS):
        total_return, max_drawdown, total_trades = run_continuous_backtest(
            price_data, sentiment_data, buy_th, and_sell_th, or_sell_th
        )

        result = {
            'buy_threshold': buy_th,
            'and_sell_threshold': and_sell_th,
            'or_sell_threshold': or_sell_th,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades
        }
        results.append(result)

    return pd.DataFrame(results)


def main():
    print("=" * 100)
    print("实验F: cycle_index 连续测试专用优化")
    print("=" * 100)
    print(f"\n测试方式: 连续5年 (2021年初$100k → 2025年末)")
    print(f"仓位: 80% 动态复利")
    print(f"\n搜索范围 (针对连续测试调整):")
    print(f"  买入阈值:      {BUY_THRESHOLDS}")
    print(f"  卖出阈值1(AND): {AND_SELL_THRESHOLDS}")
    print(f"  卖出阈值2(OR):  {OR_SELL_THRESHOLDS}")

    total_combinations = len(BUY_THRESHOLDS) * len(AND_SELL_THRESHOLDS) * len(OR_SELL_THRESHOLDS)
    print(f"  每只股票组合数: {total_combinations}")

    all_best_results = []

    for symbol in SYMBOLS:
        print(f"\n{'='*60}")
        print(f"优化 {symbol}...")
        print(f"{'='*60}")

        sentiment_data = load_cycle_index(db_config, symbol)
        price_data = load_price_with_ma(db_config, symbol)

        results_df = optimize_single_stock(symbol, sentiment_data, price_data)
        results_df = results_df.sort_values('total_return', ascending=False)

        best = results_df.iloc[0]

        print(f"\n【{symbol} 最优参数】")
        print(f"  买入: 情绪 < {best['buy_threshold']}")
        print(f"  卖出: (情绪 > {best['and_sell_threshold']} AND 价格 < MA50) OR (情绪 > {best['or_sell_threshold']})")
        print(f"  ─────────────────────────────────────────")
        print(f"  5年连续收益: {best['total_return']:+.1f}%")
        print(f"  最大回撤: {best['max_drawdown']:.1f}%")
        print(f"  交易次数: {int(best['total_trades'])}")

        best_result = {
            'symbol': symbol,
            'buy_threshold': best['buy_threshold'],
            'and_sell_threshold': best['and_sell_threshold'],
            'or_sell_threshold': best['or_sell_threshold'],
            'total_return': best['total_return'],
            'max_drawdown': best['max_drawdown'],
            'total_trades': best['total_trades']
        }
        all_best_results.append(best_result)

    # 汇总表
    print("\n" + "=" * 100)
    print("七姐妹连续测试最优参数汇总 (按5年收益排序)")
    print("=" * 100)

    summary_df = pd.DataFrame(all_best_results)
    summary_df = summary_df.sort_values('total_return', ascending=False)

    print(f"\n{'股票':<6} {'买入':<6} {'AND':<5} {'OR':<5} {'5年收益':>10} {'回撤':>8} {'交易':>6}")
    print("-" * 55)
    for _, row in summary_df.iterrows():
        print(f"{row['symbol']:<6} <{row['buy_threshold']:<4} >{row['and_sell_threshold']:<4} >{row['or_sell_threshold']:<4} "
              f"{row['total_return']:>+9.0f}% {row['max_drawdown']:>7.1f}% {int(row['total_trades']):>5}")

    print("-" * 55)
    avg_return = summary_df['total_return'].mean()
    avg_drawdown = summary_df['max_drawdown'].mean()
    print(f"{'平均':<6} {'':14} {avg_return:>+9.0f}% {avg_drawdown:>7.1f}%")

    # 与年度独立对比
    print("\n" + "=" * 100)
    print("连续测试 vs 年度独立 对比")
    print("=" * 100)

    yearly_results = {
        'NVDA': 482, 'TSLA': 625, 'AAPL': 130, 'GOOGL': 163,
        'META': 228, 'MSFT': 127, 'AMZN': 97
    }

    print(f"\n{'股票':<6} {'连续测试':>10} {'年度独立':>10} {'差异':>10}")
    print("-" * 40)
    total_cont = 0
    total_yearly = 0
    for _, row in summary_df.iterrows():
        symbol = row['symbol']
        cont_val = row['total_return']
        yearly_val = yearly_results.get(symbol, 0)
        diff = cont_val - yearly_val
        total_cont += cont_val
        total_yearly += yearly_val
        better = "✅" if diff > 0 else ""
        print(f"{symbol:<6} {cont_val:>+9.0f}% {yearly_val:>+9.0f}% {diff:>+9.0f}% {better}")

    print("-" * 40)
    print(f"{'平均':<6} {total_cont/7:>+9.0f}% {total_yearly/7:>+9.0f}% {(total_cont-total_yearly)/7:>+9.0f}%")

    # 保存结果
    output_file = f"cycle_index_continuous_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    summary_df.to_csv(output_file, index=False)
    print(f"\n结果已保存: {output_file}")


if __name__ == "__main__":
    main()
