"""
买入持有 vs 情绪策略对比
方法:
1. Buy & Hold: 年初第一个买入信号入场，持有到年底
2. 固定阈值: buy<-10, sell>30
3. 平均阈值: 4年训练取平均
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import psycopg2
from itertools import product

from data_loader import DataLoader
from backtest_engine import EnhancedBacktester

# 七姐妹股票
MAG7_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

db_config = {
    "host": "localhost",
    "port": 5432,
    "database": "crypto_fear_greed_2",
    "user": "sc2025",
    "password": ""
}

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

# 固定阈值
FIXED_BUY = -10
FIXED_SELL = 30

# 网格搜索参数
GRID_SEARCH_PARAMS = {
    "buy_thresholds": list(range(-30, 5, 5)),
    "sell_thresholds": list(range(0, 35, 5)),
}

# 测试周期
TEST_YEARS = [2021, 2022, 2023, 2024, 2025]
TRAIN_PERIODS = {
    2021: [2017, 2018, 2019, 2020],
    2022: [2018, 2019, 2020, 2021],
    2023: [2019, 2020, 2021, 2022],
    2024: [2020, 2021, 2022, 2023],
    2025: [2021, 2022, 2023, 2024],
}


def load_sentiment_data(db_config, symbol):
    conn = psycopg2.connect(**db_config)
    query = f"""
        SELECT date, smoothed_index
        FROM fear_greed_index
        WHERE symbol = '{symbol}'
        ORDER BY date
    """
    df = pd.read_sql(query, conn)
    conn.close()
    if len(df) == 0:
        return None
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.set_index('date')
    return df


def run_backtest_with_signals(stock_data, signals_df, start_date, end_date):
    """通用回测函数"""
    mask = (stock_data.index >= start_date) & (stock_data.index <= end_date)
    price_data = stock_data[mask]

    if len(price_data) == 0:
        return None, None, None

    signals = signals_df.reindex(price_data.index)
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

    portfolio, metrics, trades = backtester.run_backtest_with_sell_signal(signals, price_data)
    return portfolio, metrics, trades


def strategy_buy_hold(stock_data, sentiment_data, buy_threshold, year):
    """买入持有策略: 第一个买入信号入场，持有到年底"""
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    mask = (stock_data.index >= start_date) & (stock_data.index <= end_date)
    price_data = stock_data[mask]

    if len(price_data) == 0 or sentiment_data is None:
        return 0, 0, 0

    sent_data = sentiment_data.reindex(price_data.index)

    # 找第一个买入信号
    buy_signals = sent_data['smoothed_index'] < buy_threshold
    if not buy_signals.any():
        return 0, 0, 0

    first_buy_date = buy_signals[buy_signals].index[0]
    entry_price = price_data.loc[first_buy_date, 'Close']
    exit_price = price_data.iloc[-1]['Close']

    # 计算收益 (考虑80%仓位和交易成本)
    position_value = BACKTEST_PARAMS["initial_capital"] * BACKTEST_PARAMS["position_pct"]
    cost_rate = BACKTEST_PARAMS["commission_rate"] + BACKTEST_PARAMS["slippage_rate"]

    shares = position_value / (entry_price * (1 + cost_rate))
    final_value = shares * exit_price * (1 - cost_rate)
    cash = BACKTEST_PARAMS["initial_capital"] - position_value
    total_value = cash + final_value

    total_return = (total_value - BACKTEST_PARAMS["initial_capital"]) / BACKTEST_PARAMS["initial_capital"]

    # 简化的夏普和回撤计算
    holding_days = len(price_data.loc[first_buy_date:])

    return total_return, 0, 0


def strategy_fixed_threshold(stock_data, sentiment_data, year):
    """固定阈值策略"""
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    mask = (stock_data.index >= start_date) & (stock_data.index <= end_date)
    price_data = stock_data[mask]

    if len(price_data) == 0 or sentiment_data is None:
        return 0, 0, 0

    sent_data = sentiment_data.reindex(price_data.index)

    signals = pd.DataFrame(index=price_data.index)
    signals['smoothed_index'] = sent_data['smoothed_index']
    signals['buy_signal'] = (signals['smoothed_index'] < FIXED_BUY).astype(int)
    signals['sell_signal'] = (signals['smoothed_index'] > FIXED_SELL).astype(int)

    portfolio, metrics, trades = run_backtest_with_signals(stock_data, signals, start_date, end_date)

    if metrics:
        return (metrics.get('total_return', 0) or 0,
                metrics.get('sharpe_ratio', 0) or 0,
                metrics.get('max_drawdown', 0) or 0)
    return 0, 0, 0


def find_best_threshold_for_year(stock_data, sentiment_data, year):
    """找单年最优阈值"""
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    best_result = None
    best_score = -999

    for buy_th, sell_th in product(GRID_SEARCH_PARAMS["buy_thresholds"],
                                    GRID_SEARCH_PARAMS["sell_thresholds"]):
        if buy_th >= sell_th:
            continue

        mask = (stock_data.index >= start_date) & (stock_data.index <= end_date)
        price_data = stock_data[mask]

        if len(price_data) == 0 or sentiment_data is None:
            continue

        sent_data = sentiment_data.reindex(price_data.index)

        signals = pd.DataFrame(index=price_data.index)
        signals['smoothed_index'] = sent_data['smoothed_index']
        signals['buy_signal'] = (signals['smoothed_index'] < buy_th).astype(int)
        signals['sell_signal'] = (signals['smoothed_index'] > sell_th).astype(int)

        portfolio, metrics, trades = run_backtest_with_signals(stock_data, signals, start_date, end_date)

        if metrics and metrics.get('total_trades', 0) > 0:
            total_return = metrics.get('total_return', 0) or 0
            sharpe = metrics.get('sharpe_ratio', 0) or 0
            max_dd = abs(metrics.get('max_drawdown', 0) or 0)
            score = 0.4 * total_return + 0.4 * (sharpe / 3.0) - 0.2 * max_dd

            if score > best_score:
                best_score = score
                best_result = {
                    'buy_threshold': buy_th,
                    'sell_threshold': sell_th,
                }

    return best_result


def strategy_avg_threshold(stock_data, sentiment_data, year):
    """平均阈值策略"""
    train_years = TRAIN_PERIODS[year]

    # 找各年最优阈值
    yearly_best = []
    for train_year in train_years:
        best = find_best_threshold_for_year(stock_data, sentiment_data, train_year)
        if best:
            yearly_best.append(best)

    if len(yearly_best) == 0:
        return 0, 0, 0

    # 计算平均阈值
    avg_buy = np.mean([r['buy_threshold'] for r in yearly_best])
    avg_sell = np.mean([r['sell_threshold'] for r in yearly_best])
    avg_buy_rounded = round(avg_buy / 5) * 5
    avg_sell_rounded = round(avg_sell / 5) * 5

    # 测试
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    mask = (stock_data.index >= start_date) & (stock_data.index <= end_date)
    price_data = stock_data[mask]

    if len(price_data) == 0 or sentiment_data is None:
        return 0, 0, 0

    sent_data = sentiment_data.reindex(price_data.index)

    signals = pd.DataFrame(index=price_data.index)
    signals['smoothed_index'] = sent_data['smoothed_index']
    signals['buy_signal'] = (signals['smoothed_index'] < avg_buy_rounded).astype(int)
    signals['sell_signal'] = (signals['smoothed_index'] > avg_sell_rounded).astype(int)

    portfolio, metrics, trades = run_backtest_with_signals(stock_data, signals, start_date, end_date)

    if metrics:
        return (metrics.get('total_return', 0) or 0,
                metrics.get('sharpe_ratio', 0) or 0,
                metrics.get('max_drawdown', 0) or 0)
    return 0, 0, 0


def test_symbol(symbol, loader):
    """测试单个股票的三种策略"""
    print(f"\n{'='*70}")
    print(f"股票: {symbol}")
    print(f"{'='*70}")

    # 加载数据
    try:
        stock_data = loader.load_ohlcv(symbol, start_date="2014-01-01")
        if len(stock_data) == 0:
            print(f"  无价格数据")
            return None
    except:
        print(f"  加载价格失败")
        return None

    sentiment_data = load_sentiment_data(db_config, symbol)
    if sentiment_data is None:
        print(f"  无情绪数据")
        return None

    # 三种策略结果
    results = {
        'buy_hold': [],
        'fixed': [],
        'avg': []
    }

    print(f"\n  {'年份':<6} {'买入持有':>12} {'固定阈值':>12} {'平均阈值':>12}")
    print(f"  {'-'*50}")

    for year in TEST_YEARS:
        # 买入持有 (使用-10作为买入阈值)
        bh_ret, _, _ = strategy_buy_hold(stock_data, sentiment_data, FIXED_BUY, year)
        results['buy_hold'].append(bh_ret)

        # 固定阈值
        fixed_ret, _, _ = strategy_fixed_threshold(stock_data, sentiment_data, year)
        results['fixed'].append(fixed_ret)

        # 平均阈值
        avg_ret, _, _ = strategy_avg_threshold(stock_data, sentiment_data, year)
        results['avg'].append(avg_ret)

        print(f"  {year:<6} {bh_ret:>12.2%} {fixed_ret:>12.2%} {avg_ret:>12.2%}")

    # 计算累计收益
    bh_cum = np.prod([1 + r for r in results['buy_hold']]) - 1
    fixed_cum = np.prod([1 + r for r in results['fixed']]) - 1
    avg_cum = np.prod([1 + r for r in results['avg']]) - 1

    print(f"  {'-'*50}")
    print(f"  {'累计':<6} {bh_cum:>12.2%} {fixed_cum:>12.2%} {avg_cum:>12.2%}")

    # 年化收益
    bh_ann = (1 + bh_cum) ** (1/5) - 1
    fixed_ann = (1 + fixed_cum) ** (1/5) - 1
    avg_ann = (1 + avg_cum) ** (1/5) - 1

    print(f"  {'年化':<6} {bh_ann:>12.2%} {fixed_ann:>12.2%} {avg_ann:>12.2%}")

    # 找最优策略
    best_strategy = max([
        ('买入持有', bh_cum),
        ('固定阈值', fixed_cum),
        ('平均阈值', avg_cum)
    ], key=lambda x: x[1])
    print(f"\n  最优策略: {best_strategy[0]}")

    return {
        'symbol': symbol,
        'buy_hold_cum': bh_cum,
        'fixed_cum': fixed_cum,
        'avg_cum': avg_cum,
        'buy_hold_ann': bh_ann,
        'fixed_ann': fixed_ann,
        'avg_ann': avg_ann,
        'best': best_strategy[0]
    }


def main():
    print("=" * 70)
    print("三策略对比: 买入持有 vs 固定阈值 vs 平均阈值")
    print("=" * 70)
    print("\n策略说明:")
    print("  1. 买入持有: smoothed_index < -10 时买入，持有到年底")
    print("  2. 固定阈值: buy < -10, sell > 30")
    print("  3. 平均阈值: 4年训练取平均阈值")

    loader = DataLoader(db_config)

    all_results = []
    for symbol in MAG7_SYMBOLS:
        result = test_symbol(symbol, loader)
        if result:
            all_results.append(result)

    loader.close()

    # 总汇总
    print("\n" + "=" * 70)
    print("七姐妹汇总对比")
    print("=" * 70)

    print(f"\n{'股票':<8} {'买入持有':>12} {'固定阈值':>12} {'平均阈值':>12} {'最优':>10}")
    print("-" * 60)

    for r in sorted(all_results, key=lambda x: max(x['buy_hold_cum'], x['fixed_cum'], x['avg_cum']), reverse=True):
        print(f"{r['symbol']:<8} {r['buy_hold_cum']:>12.2%} {r['fixed_cum']:>12.2%} "
              f"{r['avg_cum']:>12.2%} {r['best']:>10}")

    print("-" * 60)

    # 平均
    avg_bh = np.mean([r['buy_hold_cum'] for r in all_results])
    avg_fixed = np.mean([r['fixed_cum'] for r in all_results])
    avg_avg = np.mean([r['avg_cum'] for r in all_results])

    print(f"{'平均':<8} {avg_bh:>12.2%} {avg_fixed:>12.2%} {avg_avg:>12.2%}")

    # 统计最优策略
    print("\n" + "=" * 70)
    print("最优策略统计")
    print("=" * 70)

    best_counts = {'买入持有': 0, '固定阈值': 0, '平均阈值': 0}
    for r in all_results:
        best_counts[r['best']] += 1

    for strategy, count in sorted(best_counts.items(), key=lambda x: x[1], reverse=True):
        stocks = [r['symbol'] for r in all_results if r['best'] == strategy]
        print(f"  {strategy}: {count}个股票 - {', '.join(stocks)}")

    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
