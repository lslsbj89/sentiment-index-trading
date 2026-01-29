"""
AND条件止损策略详细年度表现
显示每年的收益率、回撤、夏普率
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import psycopg2

from data_loader import DataLoader

MAG7_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

db_config = {
    "host": "localhost",
    "port": 5432,
    "database": "crypto_fear_greed_2",
    "user": "sc2025",
    "password": ""
}

INITIAL_CAPITAL = 100000
COMMISSION_RATE = 0.001
SLIPPAGE_RATE = 0.001
POSITION_PCT = 0.8

BUY_THRESHOLD = -10
MA_PERIOD = 50
STOP_SENTIMENT = 10

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


def run_backtest_and_condition(stock_data, sentiment_data, year):
    """AND条件止损回测，返回详细指标"""
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    lookback_start = pd.to_datetime(start_date, utc=True) - pd.Timedelta(days=MA_PERIOD * 2)
    end_dt = pd.to_datetime(end_date, utc=True)
    mask = (stock_data.index >= lookback_start) & (stock_data.index <= end_dt)
    price_data = stock_data[mask].copy()

    if len(price_data) < MA_PERIOD:
        return None

    price_data['MA'] = price_data['Close'].rolling(window=MA_PERIOD).mean()

    start_dt = pd.to_datetime(start_date, utc=True)
    test_mask = price_data.index >= start_dt
    price_data = price_data[test_mask].copy()

    if len(price_data) == 0 or sentiment_data is None:
        return None

    sent_data = sentiment_data.reindex(price_data.index)

    cash = INITIAL_CAPITAL
    shares = 0
    entry_price = 0
    entry_date = None
    trades = []
    portfolio_values = []
    daily_returns = []

    prev_value = INITIAL_CAPITAL

    for i, (date, row) in enumerate(price_data.iterrows()):
        price = row['Close']
        ma = row['MA']
        sentiment = sent_data.loc[date, 'smoothed_index'] if date in sent_data.index else None

        if pd.isna(sentiment) or pd.isna(ma):
            current_value = cash + shares * price
            portfolio_values.append(current_value)
            daily_returns.append((current_value - prev_value) / prev_value if prev_value > 0 else 0)
            prev_value = current_value
            continue

        if shares > 0:
            below_ma = price < ma
            high_sentiment = sentiment > STOP_SENTIMENT

            if below_ma and high_sentiment:
                sell_price = price * (1 - SLIPPAGE_RATE) * (1 - COMMISSION_RATE)
                cash += shares * sell_price
                trade_return = (sell_price - entry_price) / entry_price
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': date,
                    'return': trade_return,
                    'exit_reason': 'and_stop'
                })
                shares = 0
                entry_price = 0

        elif shares == 0 and sentiment < BUY_THRESHOLD:
            position_value = cash * POSITION_PCT
            buy_price = price * (1 + SLIPPAGE_RATE) * (1 + COMMISSION_RATE)
            shares = int(position_value / buy_price)
            if shares > 0:
                cash -= shares * buy_price
                entry_price = buy_price
                entry_date = date

        current_value = cash + shares * price
        portfolio_values.append(current_value)
        daily_returns.append((current_value - prev_value) / prev_value if prev_value > 0 else 0)
        prev_value = current_value

    # 年底平仓
    if shares > 0:
        final_price = price_data.iloc[-1]['Close']
        sell_price = final_price * (1 - SLIPPAGE_RATE) * (1 - COMMISSION_RATE)
        cash += shares * sell_price
        trade_return = (sell_price - entry_price) / entry_price
        trades.append({
            'entry_date': entry_date,
            'exit_date': price_data.index[-1],
            'return': trade_return,
            'exit_reason': 'year_end'
        })

    final_value = cash
    total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL

    # 最大回撤
    portfolio_series = pd.Series(portfolio_values)
    rolling_max = portfolio_series.expanding().max()
    drawdowns = (portfolio_series - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()

    # 夏普率 (假设无风险利率为0，年化)
    daily_returns_arr = np.array(daily_returns)
    if len(daily_returns_arr) > 0 and np.std(daily_returns_arr) > 0:
        sharpe = np.mean(daily_returns_arr) / np.std(daily_returns_arr) * np.sqrt(252)
    else:
        sharpe = 0

    # 止损触发次数
    and_stops = sum(1 for t in trades if t['exit_reason'] == 'and_stop')

    return {
        'year': year,
        'return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'trades': len(trades),
        'and_stops': and_stops
    }


def main():
    print("=" * 90)
    print("AND条件止损策略 - 详细年度表现")
    print("=" * 90)
    print(f"\n策略: 买入 smoothed_index < {BUY_THRESHOLD}")
    print(f"       止损 价格 < MA{MA_PERIOD} AND 情绪 > {STOP_SENTIMENT}")

    loader = DataLoader(db_config)

    all_results = {}

    for symbol in MAG7_SYMBOLS:
        stock_data = loader.load_ohlcv(symbol, start_date="2014-01-01")
        sentiment_data = load_sentiment_data(db_config, symbol)

        if stock_data is None or len(stock_data) == 0 or sentiment_data is None:
            continue

        results = []
        for year in TEST_YEARS:
            r = run_backtest_and_condition(stock_data, sentiment_data, year)
            if r:
                results.append(r)

        all_results[symbol] = results

    loader.close()

    # 打印每个股票的详细结果
    for symbol, results in all_results.items():
        print(f"\n{'='*90}")
        print(f"股票: {symbol}")
        print(f"{'='*90}")
        print(f"\n  {'年份':<6} {'收益率':>10} {'最大回撤':>10} {'夏普率':>10} {'交易':>6} {'止损触发':>8}")
        print(f"  {'-'*55}")

        for r in results:
            print(f"  {r['year']:<6} {r['return']:>10.2%} {r['max_drawdown']:>10.2%} "
                  f"{r['sharpe']:>10.2f} {r['trades']:>6} {r['and_stops']:>8}")

        # 汇总
        cum = np.prod([1 + r['return'] for r in results]) - 1
        ann = (1 + cum) ** (1/5) - 1
        avg_ret = np.mean([r['return'] for r in results])
        avg_dd = np.mean([r['max_drawdown'] for r in results])
        avg_sharpe = np.mean([r['sharpe'] for r in results])
        total_stops = sum(r['and_stops'] for r in results)

        print(f"  {'-'*55}")
        print(f"  {'平均':<6} {avg_ret:>10.2%} {avg_dd:>10.2%} {avg_sharpe:>10.2f} {'':<6} {total_stops:>8}")
        print(f"  {'累计':<6} {cum:>10.2%}")
        print(f"  {'年化':<6} {ann:>10.2%}")

    # 汇总表
    print("\n" + "=" * 90)
    print("七姐妹汇总对比")
    print("=" * 90)

    print(f"\n{'股票':<8} {'平均收益':>10} {'平均回撤':>10} {'平均夏普':>10} {'5年累计':>12} {'年化':>10} {'止损次数':>8}")
    print("-" * 80)

    summary = []
    for symbol, results in all_results.items():
        cum = np.prod([1 + r['return'] for r in results]) - 1
        ann = (1 + cum) ** (1/5) - 1
        avg_ret = np.mean([r['return'] for r in results])
        avg_dd = np.mean([r['max_drawdown'] for r in results])
        avg_sharpe = np.mean([r['sharpe'] for r in results])
        total_stops = sum(r['and_stops'] for r in results)

        summary.append({
            'symbol': symbol,
            'avg_return': avg_ret,
            'avg_drawdown': avg_dd,
            'avg_sharpe': avg_sharpe,
            'cumulative': cum,
            'annualized': ann,
            'stops': total_stops
        })

    for s in sorted(summary, key=lambda x: x['cumulative'], reverse=True):
        print(f"{s['symbol']:<8} {s['avg_return']:>10.2%} {s['avg_drawdown']:>10.2%} "
              f"{s['avg_sharpe']:>10.2f} {s['cumulative']:>12.2%} {s['annualized']:>10.2%} {s['stops']:>8}")

    print("-" * 80)
    avg_all = np.mean([s['avg_return'] for s in summary])
    avg_dd_all = np.mean([s['avg_drawdown'] for s in summary])
    avg_sharpe_all = np.mean([s['avg_sharpe'] for s in summary])
    avg_cum = np.mean([s['cumulative'] for s in summary])
    print(f"{'平均':<8} {avg_all:>10.2%} {avg_dd_all:>10.2%} {avg_sharpe_all:>10.2f} {avg_cum:>12.2%}")

    # 年度横向对比
    print("\n" + "=" * 90)
    print("年度收益率对比")
    print("=" * 90)

    header = f"{'股票':<8}"
    for year in TEST_YEARS:
        header += f" {year:>10}"
    header += f" {'累计':>12}"
    print(header)
    print("-" * 75)

    for symbol, results in all_results.items():
        row = f"{symbol:<8}"
        for r in results:
            row += f" {r['return']:>10.2%}"
        cum = np.prod([1 + r['return'] for r in results]) - 1
        row += f" {cum:>12.2%}"
        print(row)

    # 年度夏普率对比
    print("\n" + "=" * 90)
    print("年度夏普率对比")
    print("=" * 90)

    header = f"{'股票':<8}"
    for year in TEST_YEARS:
        header += f" {year:>10}"
    header += f" {'平均':>12}"
    print(header)
    print("-" * 75)

    for symbol, results in all_results.items():
        row = f"{symbol:<8}"
        for r in results:
            row += f" {r['sharpe']:>10.2f}"
        avg_sharpe = np.mean([r['sharpe'] for r in results])
        row += f" {avg_sharpe:>12.2f}"
        print(row)

    # 年度最大回撤对比
    print("\n" + "=" * 90)
    print("年度最大回撤对比")
    print("=" * 90)

    header = f"{'股票':<8}"
    for year in TEST_YEARS:
        header += f" {year:>10}"
    header += f" {'平均':>12}"
    print(header)
    print("-" * 75)

    for symbol, results in all_results.items():
        row = f"{symbol:<8}"
        for r in results:
            row += f" {r['max_drawdown']:>10.2%}"
        avg_dd = np.mean([r['max_drawdown'] for r in results])
        row += f" {avg_dd:>12.2%}"
        print(row)

    print("\n" + "=" * 90)
    print("完成!")
    print("=" * 90)


if __name__ == "__main__":
    main()
