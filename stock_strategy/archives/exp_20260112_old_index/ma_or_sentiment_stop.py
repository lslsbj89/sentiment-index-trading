"""
均线 OR 情绪止损策略测试
卖出条件: 价格下穿50日均线 OR 情绪指数 > threshold

对比 AND 条件和 OR 条件的效果
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
MA_PERIOD = 50

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


def run_backtest(stock_data, sentiment_data, year, buy_threshold,
                 ma_period, sell_sentiment, use_or=True):
    """
    回测函数
    use_or=True: 价格<MA OR 情绪>threshold
    use_or=False: 价格<MA AND 情绪>threshold
    """
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    # 需要更早的数据来计算均线
    lookback_start = pd.to_datetime(start_date, utc=True) - pd.Timedelta(days=ma_period * 2)
    end_dt = pd.to_datetime(end_date, utc=True)
    mask = (stock_data.index >= lookback_start) & (stock_data.index <= end_dt)
    price_data = stock_data[mask].copy()

    if len(price_data) < ma_period:
        return 0, 0, [], {}

    # 计算均线
    price_data['MA'] = price_data['Close'].rolling(window=ma_period).mean()

    # 只取测试年份
    start_dt = pd.to_datetime(start_date, utc=True)
    test_mask = price_data.index >= start_dt
    price_data = price_data[test_mask].copy()

    if len(price_data) == 0 or sentiment_data is None:
        return 0, 0, [], {}

    sent_data = sentiment_data.reindex(price_data.index)

    # 初始化
    cash = INITIAL_CAPITAL
    shares = 0
    entry_price = 0
    entry_date = None
    trades = []
    portfolio_values = []
    exit_reasons = {'ma': 0, 'sentiment': 0, 'year_end': 0}

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

            below_ma = price < ma
            high_sentiment = sentiment > sell_sentiment

            if use_or:
                # OR 条件
                if below_ma:
                    sell_signal = True
                    exit_reason = f'ma_stop (price<MA{ma_period})'
                    exit_reasons['ma'] += 1
                elif high_sentiment:
                    sell_signal = True
                    exit_reason = f'sentiment ({sentiment:.0f})'
                    exit_reasons['sentiment'] += 1
            else:
                # AND 条件
                if below_ma and high_sentiment:
                    sell_signal = True
                    exit_reason = f'ma+sent (price<MA, sent={sentiment:.0f})'
                    exit_reasons['ma'] += 1

            if sell_signal:
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
        exit_reasons['year_end'] += 1

    final_value = cash
    total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL

    portfolio_series = pd.Series(portfolio_values)
    rolling_max = portfolio_series.expanding().max()
    drawdowns = (portfolio_series - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()

    return total_return, max_drawdown, trades, exit_reasons


def compare_or_vs_and(loader):
    """比较 OR 和 AND 条件"""
    print("\n" + "=" * 80)
    print("OR 条件 vs AND 条件 vs 无止损 对比")
    print("=" * 80)
    print("\nOR: 价格<MA50 OR 情绪>15")
    print("AND: 价格<MA50 AND 情绪>10")
    print("无止损: 只在情绪>30时卖出")

    sell_sentiment_or = 15  # OR条件用较高阈值
    sell_sentiment_and = 10  # AND条件用较低阈值

    results = {}

    for symbol in MAG7_SYMBOLS:
        stock_data = loader.load_ohlcv(symbol, start_date="2014-01-01")
        sentiment_data = load_sentiment_data(db_config, symbol)

        if stock_data is None or len(stock_data) == 0 or sentiment_data is None:
            continue

        # 测试三种策略
        or_cum = 1.0
        and_cum = 1.0
        no_stop_cum = 1.0

        or_exits = {'ma': 0, 'sentiment': 0, 'year_end': 0}
        and_exits = {'ma': 0, 'sentiment': 0, 'year_end': 0}

        for year in TEST_YEARS:
            # OR 条件
            ret_or, _, _, exits_or = run_backtest(
                stock_data, sentiment_data, year,
                BUY_THRESHOLD, MA_PERIOD, sell_sentiment_or, use_or=True
            )
            or_cum *= (1 + ret_or)
            for k, v in exits_or.items():
                or_exits[k] += v

            # AND 条件
            ret_and, _, _, exits_and = run_backtest(
                stock_data, sentiment_data, year,
                BUY_THRESHOLD, MA_PERIOD, sell_sentiment_and, use_or=False
            )
            and_cum *= (1 + ret_and)
            for k, v in exits_and.items():
                and_exits[k] += v

            # 无止损 (情绪>30)
            ret_no, _, _, _ = run_backtest(
                stock_data, sentiment_data, year,
                BUY_THRESHOLD, MA_PERIOD, 30, use_or=False
            )
            no_stop_cum *= (1 + ret_no)

        results[symbol] = {
            'or': or_cum - 1,
            'and': and_cum - 1,
            'no_stop': no_stop_cum - 1,
            'or_exits': or_exits,
            'and_exits': and_exits
        }

    # 打印结果
    print(f"\n{'股票':<8} {'OR条件':>12} {'AND条件':>12} {'无止损':>12} {'最优':>10}")
    print("-" * 60)

    for symbol, r in results.items():
        best = max([('OR', r['or']), ('AND', r['and']), ('无止损', r['no_stop'])], key=lambda x: x[1])
        print(f"{symbol:<8} {r['or']:>12.0%} {r['and']:>12.0%} {r['no_stop']:>12.0%} {best[0]:>10}")

    print("-" * 60)
    avg_or = np.mean([r['or'] for r in results.values()])
    avg_and = np.mean([r['and'] for r in results.values()])
    avg_no = np.mean([r['no_stop'] for r in results.values()])
    print(f"{'平均':<8} {avg_or:>12.0%} {avg_and:>12.0%} {avg_no:>12.0%}")

    # 打印卖出触发统计
    print("\n" + "=" * 80)
    print("卖出触发次数统计 (5年总计)")
    print("=" * 80)

    print(f"\n{'股票':<8} {'--- OR条件 ---':^24} {'--- AND条件 ---':^24}")
    print(f"{'':8} {'MA':>8} {'情绪':>8} {'年末':>8} {'MA+情绪':>12} {'年末':>12}")
    print("-" * 60)

    for symbol, r in results.items():
        or_e = r['or_exits']
        and_e = r['and_exits']
        print(f"{symbol:<8} {or_e['ma']:>8} {or_e['sentiment']:>8} {or_e['year_end']:>8} "
              f"{and_e['ma']:>12} {and_e['year_end']:>12}")

    return results


def test_different_or_thresholds(loader):
    """测试不同 OR 情绪阈值"""
    print("\n" + "=" * 80)
    print("不同 OR 情绪阈值对比 (价格<MA50 OR 情绪>X)")
    print("=" * 80)

    thresholds = [10, 15, 20, 25, 30]

    all_results = {}

    for symbol in MAG7_SYMBOLS:
        stock_data = loader.load_ohlcv(symbol, start_date="2014-01-01")
        sentiment_data = load_sentiment_data(db_config, symbol)

        if stock_data is None or len(stock_data) == 0 or sentiment_data is None:
            continue

        symbol_results = {}
        for th in thresholds:
            cum_return = 1.0
            for year in TEST_YEARS:
                ret, _, _, _ = run_backtest(
                    stock_data, sentiment_data, year,
                    BUY_THRESHOLD, MA_PERIOD, th, use_or=True
                )
                cum_return *= (1 + ret)
            symbol_results[th] = cum_return - 1

        all_results[symbol] = symbol_results

    # 打印对比表
    print(f"\n{'股票':<8}", end="")
    for th in thresholds:
        print(f"{'>'+str(th):>10}", end="")
    print(f"{'最优':>10}")
    print("-" * 70)

    for symbol, results in all_results.items():
        print(f"{symbol:<8}", end="")
        best = max(results.items(), key=lambda x: x[1])
        for th in thresholds:
            val = results[th]
            if th == best[0]:
                print(f"**{val:>7.0%}**", end="")
            else:
                print(f"{val:>10.0%}", end="")
        print(f"{'>' + str(best[0]):>10}")

    print("-" * 70)
    print(f"{'平均':<8}", end="")
    for th in thresholds:
        avg = np.mean([results[th] for results in all_results.values()])
        print(f"{avg:>10.0%}", end="")
    print()

    return all_results


def main():
    print("=" * 80)
    print("OR 条件止损策略测试")
    print("=" * 80)
    print("\n对比:")
    print("  OR:  价格 < MA50 OR 情绪 > threshold")
    print("  AND: 价格 < MA50 AND 情绪 > threshold")

    loader = DataLoader(db_config)

    # 对比 OR vs AND vs 无止损
    compare_or_vs_and(loader)

    # 测试不同 OR 阈值
    test_different_or_thresholds(loader)

    loader.close()

    print("\n" + "=" * 80)
    print("完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
