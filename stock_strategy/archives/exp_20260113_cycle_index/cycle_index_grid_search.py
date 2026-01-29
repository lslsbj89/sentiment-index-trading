"""
实验C: cycle_index 网格搜索验证
使用新指数(cycle_index)在TSLA上测试不同阈值组合
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

# 网格搜索参数
BUY_THRESHOLDS = [-20, -15, -10, -5, 0, 5]
AND_SELL_THRESHOLDS = [15, 20, 25, 30]
OR_THRESHOLD = 50  # 固定OR兜底阈值

# 回测参数 (无止盈止损)
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
    """加载新的 cycle_index 数据"""
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
    """加载价格数据并计算MA50"""
    loader = DataLoader(db_config)
    ohlcv = loader.load_ohlcv(symbol, start_date="2015-01-01")
    loader.close()

    ohlcv['MA50'] = ohlcv['Close'].rolling(window=50).mean()
    return ohlcv


def run_single_backtest(price_data, sentiment_data, buy_threshold, and_sell_threshold,
                        or_threshold, start_date, end_date, use_dynamic=True):
    """运行单次回测"""
    mask = (price_data.index >= start_date) & (price_data.index <= end_date)
    test_price = price_data[mask].copy()
    test_sentiment = sentiment_data.reindex(test_price.index)

    if len(test_price) == 0:
        return None, None, None

    # 构建信号
    signals = pd.DataFrame(index=test_price.index)
    signals['smoothed_index'] = test_sentiment['smoothed_index']
    signals['Close'] = test_price['Close']
    signals['MA50'] = test_price['MA50']

    # 买入信号: 情绪 < buy_threshold
    signals['buy_signal'] = (signals['smoothed_index'] < buy_threshold).astype(int)

    # 卖出信号: (情绪 > and_sell_threshold AND 价格 < MA50) OR (情绪 > or_threshold)
    and_condition = (signals['smoothed_index'] > and_sell_threshold) & (signals['Close'] < signals['MA50'])
    or_condition = signals['smoothed_index'] > or_threshold
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
        use_dynamic_position=use_dynamic,
        position_pct=BACKTEST_PARAMS["position_pct"]
    )

    portfolio, metrics, trades = backtester.run_backtest_with_sell_signal(signals, test_price)
    return portfolio, metrics, trades


def run_continuous_5year_backtest(price_data, sentiment_data, buy_threshold, and_sell_threshold, or_threshold):
    """运行连续5年回测 (2021-2025)"""
    start_date = pd.Timestamp("2021-01-01", tz='UTC')
    end_date = pd.Timestamp("2025-12-31", tz='UTC')

    portfolio, metrics, trades = run_single_backtest(
        price_data, sentiment_data,
        buy_threshold, and_sell_threshold, or_threshold,
        start_date, end_date, use_dynamic=True
    )

    return portfolio, metrics, trades


def main():
    print("=" * 70)
    print(f"实验C: {SYMBOL} cycle_index 网格搜索验证")
    print("=" * 70)
    print(f"\n新指数表: cycle_index")
    print(f"买入阈值: {BUY_THRESHOLDS}")
    print(f"AND卖出阈值: {AND_SELL_THRESHOLDS}")
    print(f"OR兜底阈值: {OR_THRESHOLD}")

    # 加载数据
    print("\n加载数据...")
    sentiment_data = load_cycle_index(db_config, SYMBOL)
    price_data = load_price_with_ma(db_config, SYMBOL)
    print(f"情绪数据: {len(sentiment_data)} 行")
    print(f"价格数据: {len(price_data)} 行")

    # 网格搜索
    print("\n" + "=" * 70)
    print("开始网格搜索...")
    print("=" * 70)

    results = []

    for buy_th in BUY_THRESHOLDS:
        for and_sell_th in AND_SELL_THRESHOLDS:
            portfolio, metrics, trades = run_continuous_5year_backtest(
                price_data, sentiment_data, buy_th, and_sell_th, OR_THRESHOLD
            )

            if metrics:
                result = {
                    'buy_threshold': buy_th,
                    'and_sell_threshold': and_sell_th,
                    'or_threshold': OR_THRESHOLD,
                    'total_return': metrics.get('total_return', 0) * 100,
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0) * 100,
                    'total_trades': metrics.get('total_trades', 0),
                    'win_rate': metrics.get('win_rate', 0) * 100
                }
                results.append(result)

                print(f"buy<{buy_th:3}, sell>{and_sell_th:2} AND <MA | "
                      f"收益: {result['total_return']:7.2f}% | "
                      f"夏普: {result['sharpe_ratio']:.2f} | "
                      f"回撤: {result['max_drawdown']:.2f}% | "
                      f"交易: {result['total_trades']}")

    # 转为DataFrame并排序
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('total_return', ascending=False)

    # 输出结果
    print("\n" + "=" * 70)
    print("网格搜索结果 (按收益排序)")
    print("=" * 70)
    print(results_df.to_string(index=False))

    # 最优参数
    print("\n" + "=" * 70)
    print("最优参数组合 TOP 5")
    print("=" * 70)

    top5 = results_df.head(5)
    for i, row in top5.iterrows():
        print(f"\n#{results_df.index.get_loc(i)+1}:")
        print(f"  买入: 情绪 < {row['buy_threshold']}")
        print(f"  卖出: (情绪 > {row['and_sell_threshold']} AND 价格 < MA50) OR (情绪 > {row['or_threshold']})")
        print(f"  收益: {row['total_return']:.2f}%")
        print(f"  夏普: {row['sharpe_ratio']:.2f}")
        print(f"  回撤: {row['max_drawdown']:.2f}%")
        print(f"  交易: {int(row['total_trades'])}笔")

    # 保存结果
    output_file = f"cycle_index_grid_search_{SYMBOL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n结果已保存: {output_file}")

    # 与旧指数对比
    print("\n" + "=" * 70)
    print("与旧指数 (fear_greed_index) 最优结果对比")
    print("=" * 70)
    print(f"\n旧指数最优 (buy<-10, sell>10 AND <MA, OR>40):")
    print(f"  5年累计收益: +400% (实验20结果)")

    best = results_df.iloc[0]
    print(f"\n新指数最优 (buy<{best['buy_threshold']}, sell>{best['and_sell_threshold']} AND <MA, OR>{best['or_threshold']}):")
    print(f"  5年累计收益: +{best['total_return']:.2f}%")


if __name__ == "__main__":
    main()
