"""
实验C: cycle_index 三维网格搜索
找到 买入阈值、卖出阈值1(AND)、卖出阈值2(OR) 的最佳组合
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
AND_SELL_THRESHOLDS = [0, 5, 10, 15]  # 卖出阈值1 (AND: 价格<MA50)
OR_SELL_THRESHOLDS = [15, 20, 25, 30, 35]  # 卖出阈值2

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
    """加载 cycle_index 数据"""
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


def run_backtest(price_data, sentiment_data, buy_th, and_sell_th, or_sell_th,
                 start_date, end_date):
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
    signals['buy_signal'] = (signals['smoothed_index'] < buy_th).astype(int)

    # 卖出信号: (情绪 > and_sell_th AND 价格 < MA50) OR (情绪 > or_sell_th)
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
    return portfolio, metrics, trades


def main():
    print("=" * 80)
    print(f"实验C: {SYMBOL} cycle_index 三维网格搜索")
    print("=" * 80)
    print(f"\n策略:")
    print(f"  买入: 情绪 < 买入阈值")
    print(f"  卖出: (情绪 > 卖出阈值1 AND 价格 < MA50) OR (情绪 > 卖出阈值2)")
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
    print(f"情绪数据: {len(sentiment_data)} 行")
    print(f"价格数据: {len(price_data)} 行")

    # 回测时间范围
    start_date = pd.Timestamp("2021-01-01", tz='UTC')
    end_date = pd.Timestamp("2025-12-31", tz='UTC')

    # 三维网格搜索
    print("\n" + "=" * 80)
    print("开始网格搜索...")
    print("=" * 80)

    results = []
    count = 0

    for buy_th, and_sell_th, or_sell_th in product(BUY_THRESHOLDS, AND_SELL_THRESHOLDS, OR_SELL_THRESHOLDS):
        count += 1

        portfolio, metrics, trades = run_backtest(
            price_data, sentiment_data, buy_th, and_sell_th, or_sell_th,
            start_date, end_date
        )

        if metrics:
            result = {
                'buy_threshold': buy_th,
                'and_sell_threshold': and_sell_th,
                'or_sell_threshold': or_sell_th,
                'total_return': metrics.get('total_return', 0) * 100,
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0) * 100,
                'total_trades': metrics.get('total_trades', 0),
                'win_rate': metrics.get('win_rate', 0) * 100
            }
            results.append(result)

            # 进度显示
            if count % 20 == 0 or count == total_combinations:
                print(f"进度: {count}/{total_combinations} | "
                      f"buy<{buy_th:3}, AND>{and_sell_th:2}, OR>{or_sell_th:2} | "
                      f"收益: {result['total_return']:7.2f}%")

    # 转为DataFrame并排序
    results_df = pd.DataFrame(results)

    # 按收益排序
    results_by_return = results_df.sort_values('total_return', ascending=False)

    # 按夏普排序
    results_by_sharpe = results_df.sort_values('sharpe_ratio', ascending=False)

    # 输出结果
    print("\n" + "=" * 80)
    print("TOP 10 - 按收益排序")
    print("=" * 80)
    print(results_by_return.head(10).to_string(index=False))

    print("\n" + "=" * 80)
    print("TOP 10 - 按夏普比率排序")
    print("=" * 80)
    print(results_by_sharpe.head(10).to_string(index=False))

    # 最优参数详情
    print("\n" + "=" * 80)
    print("最优参数组合")
    print("=" * 80)

    best_return = results_by_return.iloc[0]
    best_sharpe = results_by_sharpe.iloc[0]

    print(f"\n【最高收益】")
    print(f"  买入: 情绪 < {best_return['buy_threshold']}")
    print(f"  卖出: (情绪 > {best_return['and_sell_threshold']} AND 价格 < MA50) OR (情绪 > {best_return['or_sell_threshold']})")
    print(f"  ─────────────────────────────")
    print(f"  收益: {best_return['total_return']:.2f}%")
    print(f"  夏普: {best_return['sharpe_ratio']:.2f}")
    print(f"  回撤: {best_return['max_drawdown']:.2f}%")
    print(f"  交易: {int(best_return['total_trades'])}笔")

    print(f"\n【最高夏普】")
    print(f"  买入: 情绪 < {best_sharpe['buy_threshold']}")
    print(f"  卖出: (情绪 > {best_sharpe['and_sell_threshold']} AND 价格 < MA50) OR (情绪 > {best_sharpe['or_sell_threshold']})")
    print(f"  ─────────────────────────────")
    print(f"  收益: {best_sharpe['total_return']:.2f}%")
    print(f"  夏普: {best_sharpe['sharpe_ratio']:.2f}")
    print(f"  回撤: {best_sharpe['max_drawdown']:.2f}%")
    print(f"  交易: {int(best_sharpe['total_trades'])}笔")

    # 参数敏感性分析
    print("\n" + "=" * 80)
    print("参数敏感性分析")
    print("=" * 80)

    # 按买入阈值分组
    print("\n【买入阈值影响】")
    buy_analysis = results_df.groupby('buy_threshold').agg({
        'total_return': 'mean',
        'sharpe_ratio': 'mean',
        'max_drawdown': 'mean'
    }).round(2)
    print(buy_analysis)

    # 按AND卖出阈值分组
    print("\n【AND卖出阈值影响】")
    and_analysis = results_df.groupby('and_sell_threshold').agg({
        'total_return': 'mean',
        'sharpe_ratio': 'mean',
        'max_drawdown': 'mean'
    }).round(2)
    print(and_analysis)

    # 按OR卖出阈值分组
    print("\n【OR卖出阈值影响】")
    or_analysis = results_df.groupby('or_sell_threshold').agg({
        'total_return': 'mean',
        'sharpe_ratio': 'mean',
        'max_drawdown': 'mean'
    }).round(2)
    print(or_analysis)

    # 保存结果
    output_file = f"cycle_index_3d_search_{SYMBOL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n结果已保存: {output_file}")

    # 与旧指数对比
    print("\n" + "=" * 80)
    print("与旧指数 (fear_greed_index) 对比")
    print("=" * 80)
    print(f"\n旧指数最优参数 (2021-2025):")
    print(f"  buy<-10, AND>10, OR>40 → 收益 +400%")
    print(f"\n新指数最优参数 (cycle_index):")
    print(f"  buy<{best_return['buy_threshold']}, AND>{best_return['and_sell_threshold']}, OR>{best_return['or_sell_threshold']} → 收益 +{best_return['total_return']:.2f}%")


if __name__ == "__main__":
    main()
