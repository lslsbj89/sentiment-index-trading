"""
移动止损策略测试
卖出条件: 价格从持仓期间最高点回撤 X% 时卖出

策略逻辑:
1. 买入: smoothed_index < buy_threshold
2. 卖出:
   - 价格从最高点回撤超过 trailing_stop_pct
   - 或 smoothed_index > sell_threshold (情绪卖出)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import psycopg2

from data_loader import DataLoader

# 七姐妹股票
MAG7_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

db_config = {
    "host": "localhost",
    "port": 5432,
    "database": "crypto_fear_greed_2",
    "user": "sc2025",
    "password": ""
}

# 回测参数
INITIAL_CAPITAL = 100000
COMMISSION_RATE = 0.001
SLIPPAGE_RATE = 0.001
POSITION_PCT = 0.8

# 策略参数
BUY_THRESHOLD = -10
SELL_THRESHOLD = 30  # 情绪卖出阈值
TRAILING_STOP_PCT = 0.10  # 从最高点回撤10%卖出

# 测试周期
TEST_YEARS = [2021, 2022, 2023, 2024, 2025]


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


def run_backtest_trailing_stop(stock_data, sentiment_data, year,
                                buy_threshold, sell_threshold, trailing_stop_pct):
    """带移动止损的回测"""
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    mask = (stock_data.index >= start_date) & (stock_data.index <= end_date)
    price_data = stock_data[mask].copy()

    if len(price_data) == 0 or sentiment_data is None:
        return 0, 0, []

    sent_data = sentiment_data.reindex(price_data.index)

    # 初始化
    cash = INITIAL_CAPITAL
    shares = 0
    entry_price = 0
    highest_price = 0  # 持仓期间最高价
    trades = []

    portfolio_values = []

    for i, (date, row) in enumerate(price_data.iterrows()):
        price = row['Close']
        sentiment = sent_data.loc[date, 'smoothed_index'] if date in sent_data.index else None

        if pd.isna(sentiment):
            portfolio_values.append(cash + shares * price)
            continue

        # 持有仓位时的逻辑
        if shares > 0:
            # 更新最高价
            if price > highest_price:
                highest_price = price

            # 计算从最高点的回撤
            drawdown_from_peak = (highest_price - price) / highest_price

            # 卖出条件
            sell_signal = False
            exit_reason = None

            # 条件1: 移动止损 - 从最高点回撤超过阈值
            if drawdown_from_peak >= trailing_stop_pct:
                sell_signal = True
                exit_reason = f'trailing_stop ({drawdown_from_peak:.1%})'

            # 条件2: 情绪卖出
            elif sentiment > sell_threshold:
                sell_signal = True
                exit_reason = f'sentiment ({sentiment:.1f})'

            if sell_signal:
                # 卖出
                sell_price = price * (1 - SLIPPAGE_RATE) * (1 - COMMISSION_RATE)
                cash += shares * sell_price

                trade_return = (sell_price - entry_price) / entry_price
                trades.append({
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'exit_date': date,
                    'exit_price': price,
                    'highest_price': highest_price,
                    'return': trade_return,
                    'exit_reason': exit_reason
                })

                shares = 0
                entry_price = 0
                highest_price = 0

        # 无仓位时的买入逻辑
        elif shares == 0 and sentiment < buy_threshold:
            # 买入
            position_value = cash * POSITION_PCT
            buy_price = price * (1 + SLIPPAGE_RATE) * (1 + COMMISSION_RATE)
            shares = int(position_value / buy_price)

            if shares > 0:
                cost = shares * buy_price
                cash -= cost
                entry_price = buy_price
                entry_date = date
                highest_price = price  # 初始化最高价

        portfolio_values.append(cash + shares * price)

    # 年底强制平仓
    if shares > 0:
        final_price = price_data.iloc[-1]['Close']
        sell_price = final_price * (1 - SLIPPAGE_RATE) * (1 - COMMISSION_RATE)
        cash += shares * sell_price

        trade_return = (sell_price - entry_price) / entry_price
        trades.append({
            'entry_date': entry_date,
            'entry_price': entry_price,
            'exit_date': price_data.index[-1],
            'exit_price': final_price,
            'highest_price': highest_price,
            'return': trade_return,
            'exit_reason': 'year_end'
        })

    # 计算收益
    final_value = cash
    total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL

    # 计算最大回撤
    portfolio_series = pd.Series(portfolio_values)
    rolling_max = portfolio_series.expanding().max()
    drawdowns = (portfolio_series - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()

    return total_return, max_drawdown, trades


def test_symbol(symbol, loader, trailing_stop_pct):
    """测试单个股票"""
    print(f"\n{'='*70}")
    print(f"股票: {symbol} | 移动止损: {trailing_stop_pct:.0%}")
    print(f"{'='*70}")

    # 加载数据
    try:
        stock_data = loader.load_ohlcv(symbol, start_date="2014-01-01")
        if len(stock_data) == 0:
            return None
    except:
        return None

    sentiment_data = load_sentiment_data(db_config, symbol)
    if sentiment_data is None:
        return None

    results = []
    all_trades = []

    print(f"\n  {'年份':<6} {'收益率':>10} {'回撤':>10} {'交易':>6} {'卖出原因'}")
    print(f"  {'-'*55}")

    for year in TEST_YEARS:
        ret, dd, trades = run_backtest_trailing_stop(
            stock_data, sentiment_data, year,
            BUY_THRESHOLD, SELL_THRESHOLD, trailing_stop_pct
        )

        # 统计卖出原因
        exit_reasons = {}
        for t in trades:
            reason = t['exit_reason'].split(' ')[0]  # 取主要原因
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        reason_str = ', '.join([f"{k}:{v}" for k, v in exit_reasons.items()])

        results.append({
            'year': year,
            'return': ret,
            'drawdown': dd,
            'trades': len(trades)
        })
        all_trades.extend(trades)

        print(f"  {year:<6} {ret:>10.2%} {dd:>10.2%} {len(trades):>6} {reason_str}")

    # 汇总
    cumulative = np.prod([1 + r['return'] for r in results]) - 1
    annualized = (1 + cumulative) ** (1/5) - 1
    avg_return = np.mean([r['return'] for r in results])
    avg_dd = np.mean([r['drawdown'] for r in results])

    print(f"  {'-'*55}")
    print(f"  {'累计':<6} {cumulative:>10.2%} {avg_dd:>10.2%}")
    print(f"  {'年化':<6} {annualized:>10.2%}")

    # 统计所有卖出原因
    exit_stats = {}
    for t in all_trades:
        reason = t['exit_reason'].split(' ')[0]
        exit_stats[reason] = exit_stats.get(reason, 0) + 1

    print(f"\n  卖出原因统计: {exit_stats}")

    return {
        'symbol': symbol,
        'cumulative': cumulative,
        'annualized': annualized,
        'avg_return': avg_return,
        'avg_drawdown': avg_dd,
        'total_trades': len(all_trades),
        'exit_stats': exit_stats
    }


def compare_trailing_stops(loader):
    """比较不同移动止损比例"""
    print("\n" + "=" * 70)
    print("不同移动止损比例对比")
    print("=" * 70)

    trailing_stops = [0.05, 0.10, 0.15, 0.20, 0.25, 1.0]  # 1.0 = 不启用

    all_results = {}

    for symbol in MAG7_SYMBOLS:
        stock_data = loader.load_ohlcv(symbol, start_date="2014-01-01")
        sentiment_data = load_sentiment_data(db_config, symbol)

        if stock_data is None or len(stock_data) == 0 or sentiment_data is None:
            continue

        symbol_results = {}
        for ts in trailing_stops:
            cum_return = 1.0
            for year in TEST_YEARS:
                ret, _, _ = run_backtest_trailing_stop(
                    stock_data, sentiment_data, year,
                    BUY_THRESHOLD, SELL_THRESHOLD, ts
                )
                cum_return *= (1 + ret)
            symbol_results[ts] = cum_return - 1

        all_results[symbol] = symbol_results

    # 打印对比表
    print(f"\n{'股票':<8}", end="")
    for ts in trailing_stops:
        label = f"{ts:.0%}" if ts < 1.0 else "无止损"
        print(f"{label:>10}", end="")
    print(f"{'最优':>10}")
    print("-" * 78)

    for symbol, results in all_results.items():
        print(f"{symbol:<8}", end="")
        best_ts = max(results.items(), key=lambda x: x[1])
        for ts in trailing_stops:
            val = results[ts]
            if ts == best_ts[0]:
                print(f"**{val:>7.0%}**", end="")
            else:
                print(f"{val:>10.0%}", end="")
        label = f"{best_ts[0]:.0%}" if best_ts[0] < 1.0 else "无止损"
        print(f"{label:>10}")

    # 平均
    print("-" * 78)
    print(f"{'平均':<8}", end="")
    for ts in trailing_stops:
        avg = np.mean([results[ts] for results in all_results.values()])
        print(f"{avg:>10.0%}", end="")
    print()

    return all_results


def main():
    print("=" * 70)
    print("移动止损策略测试")
    print("=" * 70)
    print(f"\n策略参数:")
    print(f"  买入阈值: smoothed_index < {BUY_THRESHOLD}")
    print(f"  卖出条件:")
    print(f"    1. 移动止损: 价格从最高点回撤 {TRAILING_STOP_PCT:.0%}")
    print(f"    2. 情绪卖出: smoothed_index > {SELL_THRESHOLD}")

    loader = DataLoader(db_config)

    # 测试默认移动止损
    print("\n" + "=" * 70)
    print(f"测试移动止损 = {TRAILING_STOP_PCT:.0%}")
    print("=" * 70)

    results_10 = []
    for symbol in MAG7_SYMBOLS:
        result = test_symbol(symbol, loader, TRAILING_STOP_PCT)
        if result:
            results_10.append(result)

    # 比较不同止损比例
    compare_results = compare_trailing_stops(loader)

    loader.close()

    # 最终建议
    print("\n" + "=" * 70)
    print("结论与建议")
    print("=" * 70)

    print("\n各股票最优移动止损比例:")
    for symbol, results in compare_results.items():
        best = max(results.items(), key=lambda x: x[1])
        label = f"{best[0]:.0%}" if best[0] < 1.0 else "无止损"
        print(f"  {symbol}: {label} (累计 {best[1]:.0%})")

    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
