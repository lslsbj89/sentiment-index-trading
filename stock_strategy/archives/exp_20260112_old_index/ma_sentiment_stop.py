"""
均线+情绪止损策略测试
卖出条件: 价格下穿50日均线 AND 情绪指数 > sentiment_threshold

策略逻辑:
1. 买入: smoothed_index < buy_threshold (-10)
2. 卖出:
   - 价格 < MA50 AND smoothed_index > sentiment_threshold
   - 或 smoothed_index > 30 (极度贪婪)
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
BUY_THRESHOLD = -10          # 情绪低于-10买入
SELL_SENTIMENT = 30          # 情绪高于30卖出（极度贪婪）
MA_PERIOD = 50               # 50日均线
STOP_SENTIMENT = 10          # 止损情绪阈值

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


def run_backtest_ma_stop(stock_data, sentiment_data, year,
                          buy_threshold, sell_sentiment, ma_period, stop_sentiment):
    """带均线+情绪止损的回测"""
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    # 需要更早的数据来计算均线
    lookback_start = pd.to_datetime(start_date, utc=True) - pd.Timedelta(days=ma_period * 2)
    end_dt = pd.to_datetime(end_date, utc=True)
    mask = (stock_data.index >= lookback_start) & (stock_data.index <= end_dt)
    price_data = stock_data[mask].copy()

    if len(price_data) < ma_period:
        return 0, 0, []

    # 计算均线
    price_data['MA'] = price_data['Close'].rolling(window=ma_period).mean()

    # 只取测试年份
    start_dt = pd.to_datetime(start_date, utc=True)
    test_mask = price_data.index >= start_dt
    price_data = price_data[test_mask].copy()

    if len(price_data) == 0 or sentiment_data is None:
        return 0, 0, []

    sent_data = sentiment_data.reindex(price_data.index)

    # 初始化
    cash = INITIAL_CAPITAL
    shares = 0
    entry_price = 0
    entry_date = None
    trades = []
    portfolio_values = []

    for i, (date, row) in enumerate(price_data.iterrows()):
        price = row['Close']
        ma = row['MA']
        sentiment = sent_data.loc[date, 'smoothed_index'] if date in sent_data.index else None

        if pd.isna(sentiment) or pd.isna(ma):
            portfolio_values.append(cash + shares * price)
            continue

        # 持有仓位时的逻辑
        if shares > 0:
            sell_signal = False
            exit_reason = None

            # 条件1: 均线+情绪止损 (价格下穿MA50 AND 情绪>阈值)
            if price < ma and sentiment > stop_sentiment:
                sell_signal = True
                exit_reason = f'ma_stop (price<MA{ma_period}, sent={sentiment:.0f})'

            # 条件2: 极度贪婪卖出
            elif sentiment > sell_sentiment:
                sell_signal = True
                exit_reason = f'sentiment ({sentiment:.0f})'

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
                    'return': trade_return,
                    'exit_reason': exit_reason
                })

                shares = 0
                entry_price = 0

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


def test_symbol(symbol, loader, stop_sentiment):
    """测试单个股票"""
    print(f"\n{'='*70}")
    print(f"股票: {symbol} | 止损条件: 价格<MA{MA_PERIOD} AND 情绪>{stop_sentiment}")
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

    print(f"\n  {'年份':<6} {'收益率':>10} {'回撤':>10} {'交易':>6} {'止损触发'}")
    print(f"  {'-'*55}")

    for year in TEST_YEARS:
        ret, dd, trades = run_backtest_ma_stop(
            stock_data, sentiment_data, year,
            BUY_THRESHOLD, SELL_SENTIMENT, MA_PERIOD, stop_sentiment
        )

        # 统计止损触发
        ma_stops = sum(1 for t in trades if 'ma_stop' in t['exit_reason'])

        results.append({
            'year': year,
            'return': ret,
            'drawdown': dd,
            'trades': len(trades),
            'ma_stops': ma_stops
        })
        all_trades.extend(trades)

        print(f"  {year:<6} {ret:>10.2%} {dd:>10.2%} {len(trades):>6} {ma_stops:>6}")

    # 汇总
    cumulative = np.prod([1 + r['return'] for r in results]) - 1
    annualized = (1 + cumulative) ** (1/5) - 1
    avg_dd = np.mean([r['drawdown'] for r in results])
    total_ma_stops = sum(r['ma_stops'] for r in results)

    print(f"  {'-'*55}")
    print(f"  {'累计':<6} {cumulative:>10.2%} {avg_dd:>10.2%} {len(all_trades):>6} {total_ma_stops:>6}")
    print(f"  {'年化':<6} {annualized:>10.2%}")

    return {
        'symbol': symbol,
        'cumulative': cumulative,
        'annualized': annualized,
        'avg_drawdown': avg_dd,
        'total_trades': len(all_trades),
        'ma_stops': total_ma_stops
    }


def compare_stop_sentiments(loader):
    """比较不同情绪止损阈值"""
    print("\n" + "=" * 70)
    print("不同情绪止损阈值对比 (价格<MA50 AND 情绪>X)")
    print("=" * 70)

    # 测试不同阈值: 0, 5, 10, 15, 20, 无止损(999)
    stop_thresholds = [0, 5, 10, 15, 20, 999]

    all_results = {}

    for symbol in MAG7_SYMBOLS:
        stock_data = loader.load_ohlcv(symbol, start_date="2014-01-01")
        sentiment_data = load_sentiment_data(db_config, symbol)

        if stock_data is None or len(stock_data) == 0 or sentiment_data is None:
            continue

        symbol_results = {}
        for st in stop_thresholds:
            cum_return = 1.0
            for year in TEST_YEARS:
                ret, _, _ = run_backtest_ma_stop(
                    stock_data, sentiment_data, year,
                    BUY_THRESHOLD, SELL_SENTIMENT, MA_PERIOD, st
                )
                cum_return *= (1 + ret)
            symbol_results[st] = cum_return - 1

        all_results[symbol] = symbol_results

    # 打印对比表
    print(f"\n{'股票':<8}", end="")
    for st in stop_thresholds:
        label = f">{st}" if st < 999 else "无止损"
        print(f"{label:>10}", end="")
    print(f"{'最优':>10}")
    print("-" * 78)

    for symbol, results in all_results.items():
        print(f"{symbol:<8}", end="")
        best = max(results.items(), key=lambda x: x[1])
        for st in stop_thresholds:
            val = results[st]
            if st == best[0]:
                print(f"**{val:>7.0%}**", end="")
            else:
                print(f"{val:>10.0%}", end="")
        label = f">{best[0]}" if best[0] < 999 else "无止损"
        print(f"{label:>10}")

    # 平均
    print("-" * 78)
    print(f"{'平均':<8}", end="")
    for st in stop_thresholds:
        avg = np.mean([results[st] for results in all_results.values()])
        print(f"{avg:>10.0%}", end="")
    print()

    return all_results


def compare_ma_periods(loader):
    """比较不同均线周期"""
    print("\n" + "=" * 70)
    print("不同均线周期对比 (价格<MA_X AND 情绪>10)")
    print("=" * 70)

    ma_periods = [20, 50, 100, 200]

    all_results = {}

    for symbol in MAG7_SYMBOLS:
        stock_data = loader.load_ohlcv(symbol, start_date="2014-01-01")
        sentiment_data = load_sentiment_data(db_config, symbol)

        if stock_data is None or len(stock_data) == 0 or sentiment_data is None:
            continue

        symbol_results = {}
        for ma in ma_periods:
            cum_return = 1.0
            for year in TEST_YEARS:
                ret, _, _ = run_backtest_ma_stop(
                    stock_data, sentiment_data, year,
                    BUY_THRESHOLD, SELL_SENTIMENT, ma, STOP_SENTIMENT
                )
                cum_return *= (1 + ret)
            symbol_results[ma] = cum_return - 1

        all_results[symbol] = symbol_results

    # 打印对比表
    print(f"\n{'股票':<8}", end="")
    for ma in ma_periods:
        print(f"{'MA'+str(ma):>10}", end="")
    print(f"{'最优':>10}")
    print("-" * 60)

    for symbol, results in all_results.items():
        print(f"{symbol:<8}", end="")
        best = max(results.items(), key=lambda x: x[1])
        for ma in ma_periods:
            val = results[ma]
            if ma == best[0]:
                print(f"**{val:>7.0%}**", end="")
            else:
                print(f"{val:>10.0%}", end="")
        print(f"{'MA'+str(best[0]):>10}")

    return all_results


def main():
    print("=" * 70)
    print("均线+情绪止损策略测试")
    print("=" * 70)
    print(f"\n策略参数:")
    print(f"  买入条件: smoothed_index < {BUY_THRESHOLD}")
    print(f"  卖出条件:")
    print(f"    1. 均线止损: 价格 < MA{MA_PERIOD} AND 情绪 > {STOP_SENTIMENT}")
    print(f"    2. 极度贪婪: 情绪 > {SELL_SENTIMENT}")

    loader = DataLoader(db_config)

    # 测试默认参数
    print("\n" + "=" * 70)
    print(f"测试默认参数: MA{MA_PERIOD} + 情绪>{STOP_SENTIMENT}")
    print("=" * 70)

    results_default = []
    for symbol in MAG7_SYMBOLS:
        result = test_symbol(symbol, loader, STOP_SENTIMENT)
        if result:
            results_default.append(result)

    # 对比不同情绪阈值
    compare_stop_sentiments(loader)

    # 对比不同均线周期
    compare_ma_periods(loader)

    loader.close()

    # 与无止损对比
    print("\n" + "=" * 70)
    print("与无止损策略对比")
    print("=" * 70)

    # 无止损结果 (从之前的实验)
    no_stop_results = {
        'NVDA': 514, 'TSLA': 336, 'AAPL': 128, 'GOOGL': 122,
        'MSFT': 96, 'AMZN': 47, 'META': 43
    }

    print(f"\n{'股票':<8} {'无止损':>12} {'MA50+情绪>10':>15} {'差异':>10} {'更优':>10}")
    print("-" * 60)

    for r in results_default:
        symbol = r['symbol']
        no_stop = no_stop_results.get(symbol, 0)
        ma_stop = r['cumulative'] * 100
        diff = ma_stop - no_stop
        better = "MA止损" if diff > 0 else "无止损"
        print(f"{symbol:<8} {no_stop:>12.0f}% {ma_stop:>15.0f}% {diff:>+10.0f}% {better:>10}")

    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
