"""
单股票详细分析: 分年度参数调优
记录完整交易数据：时间、价格、指数、收益率、夏普率、回撤率
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime
from itertools import product

from data_loader import DataLoader
from backtest_engine import EnhancedBacktester

# 配置
SYMBOL = "AMZN"
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
    """运行单年回测，返回详细指标和交易记录"""
    start_date = pd.Timestamp(f"{year}-01-01", tz='UTC')
    end_date = pd.Timestamp(f"{year}-12-31", tz='UTC')

    mask = (price_data.index >= start_date) & (price_data.index <= end_date)
    test_price = price_data[mask].copy()
    test_sentiment = sentiment_data.reindex(test_price.index)

    if len(test_price) == 0:
        return None, None

    # 构建信号
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

    # 增强交易记录
    if trades:
        for trade in trades:
            entry_date = trade['entry_date']
            exit_date = trade['exit_date']
            if entry_date in test_sentiment.index:
                trade['entry_index'] = test_sentiment.loc[entry_date, 'smoothed_index']
            else:
                trade['entry_index'] = np.nan
            if exit_date in test_sentiment.index:
                trade['exit_index'] = test_sentiment.loc[exit_date, 'smoothed_index']
            else:
                trade['exit_index'] = np.nan

    return metrics, trades


def main():
    print("=" * 120)
    print(f"{SYMBOL} 分年度参数调优详细分析")
    print("=" * 120)
    print(f"\n搜索范围:")
    print(f"  买入阈值:      {BUY_THRESHOLDS}")
    print(f"  卖出阈值1(AND): {AND_SELL_THRESHOLDS}")
    print(f"  卖出阈值2(OR):  {OR_SELL_THRESHOLDS}")

    total_combinations = len(BUY_THRESHOLDS) * len(AND_SELL_THRESHOLDS) * len(OR_SELL_THRESHOLDS)
    print(f"  总组合数: {total_combinations}")

    # 加载数据
    print("\n加载数据...")
    sentiment_data = load_cycle_index(db_config, SYMBOL)
    price_data = load_price_with_ma(db_config, SYMBOL)
    print(f"情绪数据: {len(sentiment_data)} 行")
    print(f"价格数据: {len(price_data)} 行")

    # 网格搜索
    results = []
    all_trades = []

    for buy_th, and_sell_th, or_sell_th in product(BUY_THRESHOLDS, AND_SELL_THRESHOLDS, OR_SELL_THRESHOLDS):
        yearly_data = {}

        for year in TEST_YEARS:
            metrics, trades = run_single_year_backtest(
                price_data, sentiment_data, buy_th, and_sell_th, or_sell_th, year
            )

            if metrics:
                yearly_data[year] = {
                    'return': metrics.get('total_return', 0) * 100,
                    'sharpe': metrics.get('sharpe_ratio', 0),
                    'drawdown': metrics.get('max_drawdown', 0) * 100,
                    'trades': metrics.get('total_trades', 0),
                    'win_rate': metrics.get('win_rate', 0) * 100
                }

                # 保存交易记录
                if trades:
                    for t in trades:
                        t['year'] = year
                        t['buy_th'] = buy_th
                        t['and_sell_th'] = and_sell_th
                        t['or_sell_th'] = or_sell_th
                        all_trades.append(t)
            else:
                yearly_data[year] = {'return': 0, 'sharpe': 0, 'drawdown': 0, 'trades': 0, 'win_rate': 0}

        # 计算汇总
        returns = [yearly_data[y]['return'] for y in TEST_YEARS]
        sharpes = [yearly_data[y]['sharpe'] for y in TEST_YEARS]
        drawdowns = [yearly_data[y]['drawdown'] for y in TEST_YEARS]

        cumulative = np.prod([1 + r/100 for r in returns]) * 100 - 100
        avg_return = np.mean(returns)
        avg_sharpe = np.mean(sharpes)
        max_drawdown = min(drawdowns)
        positive_years = sum(1 for r in returns if r > 0)

        result = {
            'buy_th': buy_th,
            'and_sell_th': and_sell_th,
            'or_sell_th': or_sell_th,
            **{f'ret_{y}': yearly_data[y]['return'] for y in TEST_YEARS},
            **{f'sharpe_{y}': yearly_data[y]['sharpe'] for y in TEST_YEARS},
            **{f'dd_{y}': yearly_data[y]['drawdown'] for y in TEST_YEARS},
            **{f'trades_{y}': yearly_data[y]['trades'] for y in TEST_YEARS},
            'cumulative': cumulative,
            'avg_return': avg_return,
            'avg_sharpe': avg_sharpe,
            'max_drawdown': max_drawdown,
            'positive_years': positive_years
        }
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('cumulative', ascending=False)

    # TOP 5 详细显示
    print("\n" + "=" * 120)
    print("TOP 5 参数组合 (按5年累计收益排序)")
    print("=" * 120)

    for rank, (_, row) in enumerate(results_df.head(5).iterrows(), 1):
        print(f"\n{'─'*120}")
        print(f"【第 {rank} 名】 买入<{row['buy_th']}, AND>{row['and_sell_th']}, OR>{row['or_sell_th']}")
        print(f"{'─'*120}")

        print(f"\n{'年份':<8} {'收益率':>10} {'夏普率':>10} {'回撤':>10} {'交易数':>8}")
        print("-" * 50)
        for year in TEST_YEARS:
            ret = row[f'ret_{year}']
            sharpe = row[f'sharpe_{year}']
            dd = row[f'dd_{year}']
            trades_count = int(row[f'trades_{year}'])
            print(f"{year:<8} {ret:>+9.1f}% {sharpe:>10.2f} {dd:>9.1f}% {trades_count:>8}")

        print("-" * 50)
        print(f"{'汇总':<8} {row['avg_return']:>+9.1f}% {row['avg_sharpe']:>10.2f} {row['max_drawdown']:>9.1f}%")
        print(f"{'累计':<8} {row['cumulative']:>+9.1f}%")
        print(f"{'盈利年':<8} {int(row['positive_years'])}/5")

        # 显示该参数组合的交易记录
        param_trades = [t for t in all_trades
                       if t['buy_th'] == row['buy_th']
                       and t['and_sell_th'] == row['and_sell_th']
                       and t['or_sell_th'] == row['or_sell_th']]

        if param_trades:
            print(f"\n交易明细:")
            print(f"{'年份':<6} {'买入日期':<12} {'买入价':>10} {'买入指数':>10} {'卖出日期':<12} {'卖出价':>10} {'卖出指数':>10} {'收益':>10} {'退出原因':<15}")
            print("-" * 110)
            for t in param_trades:
                entry_date = t['entry_date'].strftime('%Y-%m-%d') if hasattr(t['entry_date'], 'strftime') else str(t['entry_date'])[:10]
                exit_date = t['exit_date'].strftime('%Y-%m-%d') if hasattr(t['exit_date'], 'strftime') else str(t['exit_date'])[:10]
                entry_idx = t.get('entry_index', np.nan)
                exit_idx = t.get('exit_index', np.nan)
                profit_pct = t.get('profit_pct', 0) * 100

                print(f"{t['year']:<6} {entry_date:<12} {t['entry_price']:>10.2f} {entry_idx:>10.2f} "
                      f"{exit_date:<12} {t['exit_price']:>10.2f} {exit_idx:>10.2f} {profit_pct:>+9.1f}% {t['exit_reason']:<15}")

    # 保存完整结果
    output_file = f"cycle_index_{SYMBOL}_detail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n\n完整结果已保存: {output_file}")

    # 保存交易记录
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_file = f"cycle_index_{SYMBOL}_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        trades_df.to_csv(trades_file, index=False)
        print(f"交易记录已保存: {trades_file}")


if __name__ == "__main__":
    main()
