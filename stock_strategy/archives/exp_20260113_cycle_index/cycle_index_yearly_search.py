"""
实验C: cycle_index 年度独立网格搜索
每年重置资金，查看各年表现
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
SYMBOL = "TSLA"
db_config = {
    "host": "localhost",
    "port": 5432,
    "database": "crypto_fear_greed_2",
    "user": "sc2025",
    "password": ""
}

# 三维网格搜索参数
BUY_THRESHOLDS = [-25, -20, -15, -10, -5]
AND_SELL_THRESHOLDS = [0, 5, 10, 15]
OR_SELL_THRESHOLDS = [15, 20, 25, 30, 35]

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
    """运行单年回测"""
    start_date = pd.Timestamp(f"{year}-01-01", tz='UTC')
    end_date = pd.Timestamp(f"{year}-12-31", tz='UTC')

    mask = (price_data.index >= start_date) & (price_data.index <= end_date)
    test_price = price_data[mask].copy()
    test_sentiment = sentiment_data.reindex(test_price.index)

    if len(test_price) == 0:
        return None

    # 构建信号
    signals = pd.DataFrame(index=test_price.index)
    signals['smoothed_index'] = test_sentiment['smoothed_index']
    signals['Close'] = test_price['Close']
    signals['MA50'] = test_price['MA50']

    # 买入信号
    signals['buy_signal'] = (signals['smoothed_index'] < buy_th).astype(int)

    # 卖出信号: (情绪 > and_sell_th AND 价格 < MA50) OR (情绪 > or_sell_th)
    and_condition = (signals['smoothed_index'] > and_sell_th) & (signals['Close'] < signals['MA50'])
    or_condition = signals['smoothed_index'] > or_sell_th
    signals['sell_signal'] = (and_condition | or_condition).astype(int)

    signals['prob_profit'] = 0.5
    signals['position_size'] = 0

    # 回测 (每年重置资金)
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


def main():
    print("=" * 90)
    print(f"实验C: {SYMBOL} cycle_index 年度独立网格搜索")
    print("=" * 90)
    print(f"\n测试方式: 每年独立测试，年初重置资金 $100,000")
    print(f"仓位: 80% 动态复利")
    print(f"测试年份: {TEST_YEARS}")
    print(f"\n搜索范围:")
    print(f"  买入阈值:      {BUY_THRESHOLDS}")
    print(f"  卖出阈值1(AND): {AND_SELL_THRESHOLDS}")
    print(f"  卖出阈值2(OR):  {OR_SELL_THRESHOLDS}")

    total_combinations = len(BUY_THRESHOLDS) * len(AND_SELL_THRESHOLDS) * len(OR_SELL_THRESHOLDS)
    print(f"\n总组合数: {total_combinations}")

    # 加载数据
    print("\n加载数据...")
    sentiment_data = load_cycle_index(db_config, SYMBOL)
    price_data = load_price_with_ma(db_config, SYMBOL)

    # 网格搜索
    print("\n" + "=" * 90)
    print("开始网格搜索...")
    print("=" * 90)

    results = []
    count = 0

    for buy_th, and_sell_th, or_sell_th in product(BUY_THRESHOLDS, AND_SELL_THRESHOLDS, OR_SELL_THRESHOLDS):
        count += 1

        yearly_returns = {}
        for year in TEST_YEARS:
            ret = run_single_year_backtest(price_data, sentiment_data, buy_th, and_sell_th, or_sell_th, year)
            yearly_returns[year] = ret if ret else 0

        # 计算汇总指标
        returns_list = list(yearly_returns.values())
        avg_return = np.mean(returns_list)
        cumulative = np.prod([1 + r/100 for r in returns_list]) * 100 - 100  # 5年累计
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

        # 进度显示
        if count % 25 == 0 or count == total_combinations:
            print(f"进度: {count}/{total_combinations}")

    # 转为DataFrame
    results_df = pd.DataFrame(results)

    # 按累计收益排序
    results_df = results_df.sort_values('cumulative', ascending=False)

    # 输出结果
    print("\n" + "=" * 90)
    print("TOP 15 - 按5年累计收益排序")
    print("=" * 90)

    display_cols = ['buy_threshold', 'and_sell_threshold', 'or_sell_threshold',
                    'y2021', 'y2022', 'y2023', 'y2024', 'y2025', 'avg_return', 'cumulative', 'positive_years']

    top15 = results_df.head(15)[display_cols].copy()
    top15 = top15.round(1)
    print(top15.to_string(index=False))

    # 最优参数详情
    print("\n" + "=" * 90)
    print("最优参数组合")
    print("=" * 90)

    best = results_df.iloc[0]
    print(f"\n【最高累计收益】")
    print(f"  买入: 情绪 < {best['buy_threshold']}")
    print(f"  卖出: (情绪 > {best['and_sell_threshold']} AND 价格 < MA50) OR (情绪 > {best['or_sell_threshold']})")
    print(f"  ─────────────────────────────────────────")
    print(f"  2021: {best['y2021']:+.1f}%")
    print(f"  2022: {best['y2022']:+.1f}%")
    print(f"  2023: {best['y2023']:+.1f}%")
    print(f"  2024: {best['y2024']:+.1f}%")
    print(f"  2025: {best['y2025']:+.1f}%")
    print(f"  ─────────────────────────────────────────")
    print(f"  平均收益: {best['avg_return']:+.1f}%/年")
    print(f"  5年累计: {best['cumulative']:+.1f}%")
    print(f"  盈利年数: {int(best['positive_years'])}/5")

    # 稳定性最好的参数 (盈利年数最多，然后按累计收益)
    stable_df = results_df.sort_values(['positive_years', 'cumulative'], ascending=[False, False])
    stable_best = stable_df.iloc[0]

    print(f"\n【最稳定参数】(盈利年数优先)")
    print(f"  买入: 情绪 < {stable_best['buy_threshold']}")
    print(f"  卖出: (情绪 > {stable_best['and_sell_threshold']} AND 价格 < MA50) OR (情绪 > {stable_best['or_sell_threshold']})")
    print(f"  ─────────────────────────────────────────")
    print(f"  2021: {stable_best['y2021']:+.1f}%")
    print(f"  2022: {stable_best['y2022']:+.1f}%")
    print(f"  2023: {stable_best['y2023']:+.1f}%")
    print(f"  2024: {stable_best['y2024']:+.1f}%")
    print(f"  2025: {stable_best['y2025']:+.1f}%")
    print(f"  ─────────────────────────────────────────")
    print(f"  平均收益: {stable_best['avg_return']:+.1f}%/年")
    print(f"  5年累计: {stable_best['cumulative']:+.1f}%")
    print(f"  盈利年数: {int(stable_best['positive_years'])}/5")

    # 2022熊市表现最好的参数
    bear_df = results_df.sort_values('y2022', ascending=False)
    bear_best = bear_df.iloc[0]

    print(f"\n【2022熊市最佳】")
    print(f"  买入: 情绪 < {bear_best['buy_threshold']}")
    print(f"  卖出: (情绪 > {bear_best['and_sell_threshold']} AND 价格 < MA50) OR (情绪 > {bear_best['or_sell_threshold']})")
    print(f"  2022收益: {bear_best['y2022']:+.1f}%")
    print(f"  5年累计: {bear_best['cumulative']:+.1f}%")

    # 保存结果
    output_file = f"cycle_index_yearly_{SYMBOL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n结果已保存: {output_file}")


if __name__ == "__main__":
    main()
