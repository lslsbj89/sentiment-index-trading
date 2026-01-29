"""
实验D: cycle_index 七姐妹验证
用最优参数在 Magnificent 7 上测试
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
SYMBOLS = ['NVDA', 'TSLA', 'AAPL', 'GOOGL', 'META', 'MSFT', 'AMZN']
db_config = {
    "host": "localhost",
    "port": 5432,
    "database": "crypto_fear_greed_2",
    "user": "sc2025",
    "password": ""
}

# 最优参数 (来自TSLA网格搜索)
BUY_THRESHOLD = -5
AND_SELL_THRESHOLD = 15
OR_SELL_THRESHOLD = 30

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
        return None, 0

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

    # 回测
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
        return metrics, metrics.get('total_return', 0) * 100
    return None, 0


def main():
    print("=" * 100)
    print("实验D: cycle_index 七姐妹验证")
    print("=" * 100)
    print(f"\n策略参数 (来自TSLA网格搜索最优):")
    print(f"  买入: 情绪 < {BUY_THRESHOLD}")
    print(f"  卖出: (情绪 > {AND_SELL_THRESHOLD} AND 价格 < MA50) OR (情绪 > {OR_SELL_THRESHOLD})")
    print(f"\n测试方式: 每年独立测试，年初重置资金 $100,000")
    print(f"仓位: 80% 动态复利")
    print(f"测试年份: {TEST_YEARS}")
    print(f"测试股票: {SYMBOLS}")

    # 存储所有结果
    all_results = {}

    for symbol in SYMBOLS:
        print(f"\n{'='*50}")
        print(f"测试 {symbol}...")
        print(f"{'='*50}")

        # 加载数据
        try:
            sentiment_data = load_cycle_index(db_config, symbol)
            price_data = load_price_with_ma(db_config, symbol)
        except Exception as e:
            print(f"  加载数据失败: {e}")
            continue

        yearly_returns = {}
        for year in TEST_YEARS:
            metrics, ret = run_single_year_backtest(
                price_data, sentiment_data,
                BUY_THRESHOLD, AND_SELL_THRESHOLD, OR_SELL_THRESHOLD,
                year
            )
            yearly_returns[year] = ret
            print(f"  {year}: {ret:+.1f}%")

        # 计算汇总
        returns_list = list(yearly_returns.values())
        avg_return = np.mean(returns_list)
        cumulative = np.prod([1 + r/100 for r in returns_list]) * 100 - 100
        positive_years = sum(1 for r in returns_list if r > 0)

        all_results[symbol] = {
            'yearly': yearly_returns,
            'avg': avg_return,
            'cumulative': cumulative,
            'positive_years': positive_years
        }

        print(f"  ────────────────")
        print(f"  平均: {avg_return:+.1f}%/年")
        print(f"  累计: {cumulative:+.1f}%")
        print(f"  盈利: {positive_years}/5年")

    # 汇总表格
    print("\n" + "=" * 100)
    print("七姐妹汇总表 (按5年累计排序)")
    print("=" * 100)

    # 构建DataFrame
    summary_data = []
    for symbol in SYMBOLS:
        if symbol in all_results:
            r = all_results[symbol]
            row = {
                '股票': symbol,
                '2021': r['yearly'].get(2021, 0),
                '2022': r['yearly'].get(2022, 0),
                '2023': r['yearly'].get(2023, 0),
                '2024': r['yearly'].get(2024, 0),
                '2025': r['yearly'].get(2025, 0),
                '平均': r['avg'],
                '5年累计': r['cumulative'],
                '盈利年': r['positive_years']
            }
            summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('5年累计', ascending=False)

    # 格式化输出
    print(f"\n{'股票':<6} {'2021':>8} {'2022':>8} {'2023':>8} {'2024':>8} {'2025':>8} {'平均':>8} {'5年累计':>10} {'盈利年':>6}")
    print("-" * 80)
    for _, row in summary_df.iterrows():
        print(f"{row['股票']:<6} {row['2021']:>+7.1f}% {row['2022']:>+7.1f}% {row['2023']:>+7.1f}% {row['2024']:>+7.1f}% {row['2025']:>+7.1f}% {row['平均']:>+7.1f}% {row['5年累计']:>+9.1f}% {int(row['盈利年']):>5}/5")

    # 计算平均
    print("-" * 80)
    avg_row = summary_df.mean(numeric_only=True)
    print(f"{'平均':<6} {avg_row['2021']:>+7.1f}% {avg_row['2022']:>+7.1f}% {avg_row['2023']:>+7.1f}% {avg_row['2024']:>+7.1f}% {avg_row['2025']:>+7.1f}% {avg_row['平均']:>+7.1f}% {avg_row['5年累计']:>+9.1f}%")

    # 2022熊市分析
    print("\n" + "=" * 100)
    print("2022熊市表现")
    print("=" * 100)
    bear_sorted = summary_df.sort_values('2022', ascending=False)
    print(f"最佳: {bear_sorted.iloc[0]['股票']} ({bear_sorted.iloc[0]['2022']:+.1f}%)")
    print(f"最差: {bear_sorted.iloc[-1]['股票']} ({bear_sorted.iloc[-1]['2022']:+.1f}%)")

    # 与旧指数对比
    print("\n" + "=" * 100)
    print("与旧指数 (fear_greed_index) 对比")
    print("=" * 100)
    print("\n旧指数最优结果 (实验16, 年度独立):")
    print("  NVDA: +602%")
    print("  TSLA: +329%")
    print("  GOOGL: +199%")
    print("  AAPL: +125%")
    print("  META: +109%")
    print("  MSFT: +94%")
    print("  AMZN: +52%")
    print("  平均: +216%")

    new_avg = summary_df['5年累计'].mean()
    print(f"\n新指数 (cycle_index) 结果:")
    print(f"  平均: {new_avg:+.1f}%")

    # 保存结果
    output_file = f"cycle_index_mag7_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    summary_df.to_csv(output_file, index=False)
    print(f"\n结果已保存: {output_file}")


if __name__ == "__main__":
    main()
