"""
实验E: cycle_index 连续5年测试
用各股票最优参数，2021年初$100k一直持有到2025年末
与年度独立测试对比
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime

from data_loader import DataLoader
from backtest_engine import EnhancedBacktester

# 配置
db_config = {
    "host": "localhost",
    "port": 5432,
    "database": "crypto_fear_greed_2",
    "user": "sc2025",
    "password": ""
}

# 各股票最优参数 (来自年度独立优化)
OPTIMAL_PARAMS = {
    'NVDA': {'buy': -25, 'and_sell': 0, 'or_sell': 20},
    'TSLA': {'buy': -5, 'and_sell': 15, 'or_sell': 30},
    'AAPL': {'buy': -25, 'and_sell': 0, 'or_sell': 20},
    'GOOGL': {'buy': -25, 'and_sell': 0, 'or_sell': 20},
    'META': {'buy': -20, 'and_sell': 0, 'or_sell': 20},
    'MSFT': {'buy': -25, 'and_sell': 0, 'or_sell': 20},
    'AMZN': {'buy': -25, 'and_sell': 0, 'or_sell': 20},
}

# 年度独立测试结果 (用于对比)
YEARLY_RESULTS = {
    'NVDA': {'cumulative': 482, 'yearly': {2021: 31, 2022: 38, 2023: 211, 2024: 14, 2025: 20}},
    'TSLA': {'cumulative': 625, 'yearly': {2021: 91, 2022: -41, 2023: 133, 2024: 77, 2025: 56}},
    'AAPL': {'cumulative': 130, 'yearly': {2021: 34, 2022: 11, 2023: 28, 2024: 14, 2025: 17}},
    'GOOGL': {'cumulative': 163, 'yearly': {2021: 47, 2022: -1, 2023: 46, 2024: 24, 2025: 19}},
    'META': {'cumulative': 228, 'yearly': {2021: 9, 2022: 31, 2023: 72, 2024: 37, 2025: 19}},
    'MSFT': {'cumulative': 127, 'yearly': {2021: 41, 2022: 5, 2023: 35, 2024: 10, 2025: 17}},
    'AMZN': {'cumulative': 97, 'yearly': {2021: 16, 2022: 18, 2023: 34, 2024: 16, 2025: 3}},
}

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
    """运行连续5年回测 (2021-2025)"""
    start_date = pd.Timestamp("2021-01-01", tz='UTC')
    end_date = pd.Timestamp("2025-12-31", tz='UTC')

    mask = (price_data.index >= start_date) & (price_data.index <= end_date)
    test_price = price_data[mask].copy()
    test_sentiment = sentiment_data.reindex(test_price.index)

    if len(test_price) == 0:
        return None, 0, 0

    # 构建信号
    signals = pd.DataFrame(index=test_price.index)
    signals['smoothed_index'] = test_sentiment['smoothed_index']
    signals['Close'] = test_price['Close']
    signals['MA50'] = test_price['MA50']

    # 买入信号
    signals['buy_signal'] = (signals['smoothed_index'] < buy_th).astype(int)

    # 卖出信号
    and_condition = (signals['smoothed_index'] > and_sell_th) & (signals['Close'] < signals['MA50'])
    or_condition = signals['smoothed_index'] > or_sell_th
    signals['sell_signal'] = (and_condition | or_condition).astype(int)

    signals['prob_profit'] = 0.5
    signals['position_size'] = 0

    # 回测 (连续5年，不重置资金)
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
        total_return = metrics.get('total_return', 0) * 100
        max_drawdown = metrics.get('max_drawdown', 0) * 100
        return metrics, total_return, max_drawdown
    return None, 0, 0


def main():
    print("=" * 100)
    print("实验E: cycle_index 连续5年测试")
    print("=" * 100)
    print(f"\n测试方式: 2021年初 $100,000 → 2025年末 (连续5年不重置)")
    print(f"仓位: 80% 动态复利")
    print(f"测试期间: 2021-01-01 ~ 2025-12-31")

    # 存储结果
    all_results = []

    for symbol, params in OPTIMAL_PARAMS.items():
        print(f"\n{'='*60}")
        print(f"测试 {symbol}...")
        print(f"参数: buy<{params['buy']}, AND>{params['and_sell']}, OR>{params['or_sell']}")
        print(f"{'='*60}")

        # 加载数据
        try:
            sentiment_data = load_cycle_index(db_config, symbol)
            price_data = load_price_with_ma(db_config, symbol)
        except Exception as e:
            print(f"  加载数据失败: {e}")
            continue

        # 连续回测
        metrics, total_return, max_drawdown = run_continuous_backtest(
            price_data, sentiment_data,
            params['buy'], params['and_sell'], params['or_sell']
        )

        # 年度独立结果
        yearly_cumulative = YEARLY_RESULTS.get(symbol, {}).get('cumulative', 0)

        result = {
            'symbol': symbol,
            'buy_threshold': params['buy'],
            'and_sell_threshold': params['and_sell'],
            'or_sell_threshold': params['or_sell'],
            'continuous_return': total_return,
            'yearly_cumulative': yearly_cumulative,
            'max_drawdown': max_drawdown,
            'total_trades': metrics.get('total_trades', 0) if metrics else 0,
            'win_rate': metrics.get('win_rate', 0) * 100 if metrics else 0,
        }
        all_results.append(result)

        print(f"  连续5年收益: {total_return:+.1f}%")
        print(f"  年度独立累计: {yearly_cumulative:+.0f}%")
        print(f"  差异: {total_return - yearly_cumulative:+.1f}%")
        print(f"  最大回撤: {max_drawdown:.1f}%")

    # 汇总表格
    print("\n" + "=" * 100)
    print("连续5年 vs 年度独立 对比 (按连续收益排序)")
    print("=" * 100)

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('continuous_return', ascending=False)

    print(f"\n{'股票':<6} {'买入':<6} {'AND':<5} {'OR':<5} {'连续5年':>10} {'年度累计':>10} {'差异':>8} {'回撤':>8} {'交易':>6} {'胜率':>6}")
    print("-" * 85)
    for _, row in results_df.iterrows():
        diff = row['continuous_return'] - row['yearly_cumulative']
        print(f"{row['symbol']:<6} <{row['buy_threshold']:<4} >{row['and_sell_threshold']:<4} >{row['or_sell_threshold']:<4} "
              f"{row['continuous_return']:>+9.0f}% {row['yearly_cumulative']:>+9.0f}% {diff:>+7.0f}% "
              f"{row['max_drawdown']:>7.1f}% {int(row['total_trades']):>5} {row['win_rate']:>5.0f}%")

    # 平均值
    print("-" * 85)
    avg_continuous = results_df['continuous_return'].mean()
    avg_yearly = results_df['yearly_cumulative'].mean()
    avg_drawdown = results_df['max_drawdown'].mean()
    print(f"{'平均':<6} {'':14} {avg_continuous:>+9.0f}% {avg_yearly:>+9.0f}% {avg_continuous-avg_yearly:>+7.0f}% {avg_drawdown:>7.1f}%")

    # 分析
    print("\n" + "=" * 100)
    print("分析")
    print("=" * 100)

    better_count = sum(1 for _, r in results_df.iterrows() if r['continuous_return'] > r['yearly_cumulative'])
    print(f"\n连续测试更好: {better_count}/7 只股票")

    if avg_continuous > avg_yearly:
        print(f"结论: 连续测试平均收益 (+{avg_continuous:.0f}%) 高于年度独立 (+{avg_yearly:.0f}%)")
        print("原因: 动态复利在连续模式下效果更强，盈利后仓位增大")
    else:
        print(f"结论: 年度独立平均收益 (+{avg_yearly:.0f}%) 高于连续测试 (+{avg_continuous:.0f}%)")
        print("原因: 年度重置可以在大跌年后重新开始，避免复利反噬")

    # 保存结果
    output_file = f"cycle_index_continuous_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n结果已保存: {output_file}")


if __name__ == "__main__":
    main()
