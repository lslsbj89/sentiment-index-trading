"""
连续5年回测 - 动态复利 vs 固定仓位
不按年度切分，连续计算复利效果
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


def run_continuous_backtest(stock_data, sentiment_data, use_dynamic=True):
    """
    连续5年回测 (2021-2025)
    use_dynamic=True: 动态复利 (总资产 × 80%)
    use_dynamic=False: 固定仓位 (初始资金 × 80%)
    """
    start_date = "2021-01-01"
    end_date = "2025-12-31"

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
    fixed_position_value = INITIAL_CAPITAL * POSITION_PCT  # 固定仓位金额

    for i, (date, row) in enumerate(price_data.iterrows()):
        price = row['Close']
        ma = row['MA']
        sentiment = sent_data.loc[date, 'smoothed_index'] if date in sent_data.index else None

        current_total = cash + shares * price

        if pd.isna(sentiment) or pd.isna(ma):
            portfolio_values.append(current_total)
            daily_returns.append((current_total - prev_value) / prev_value if prev_value > 0 else 0)
            prev_value = current_total
            continue

        # 止损逻辑
        if shares > 0:
            below_ma = price < ma
            high_sentiment = sentiment > STOP_SENTIMENT

            if below_ma and high_sentiment:
                sell_price = price * (1 - SLIPPAGE_RATE) * (1 - COMMISSION_RATE)
                proceeds = shares * sell_price
                cash += proceeds
                trade_return = (sell_price - entry_price) / entry_price
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': date,
                    'entry_price': entry_price,
                    'exit_price': sell_price,
                    'return': trade_return,
                    'exit_reason': 'and_stop',
                    'proceeds': proceeds
                })
                shares = 0
                entry_price = 0

        # 买入逻辑
        elif shares == 0 and sentiment < BUY_THRESHOLD:
            current_total = cash  # 无持仓时，总资产=现金

            if use_dynamic:
                # 动态复利: 总资产 × 80%
                position_value = current_total * POSITION_PCT
            else:
                # 固定仓位: 初始资金 × 80%
                position_value = min(fixed_position_value, cash)

            buy_price = price * (1 + SLIPPAGE_RATE) * (1 + COMMISSION_RATE)
            shares = int(position_value / buy_price)

            if shares > 0:
                cost = shares * buy_price
                if cost <= cash:
                    cash -= cost
                    entry_price = buy_price
                    entry_date = date
                else:
                    shares = 0

        current_total = cash + shares * price
        portfolio_values.append(current_total)
        daily_returns.append((current_total - prev_value) / prev_value if prev_value > 0 else 0)
        prev_value = current_total

    # 最终清仓
    if shares > 0:
        final_price = price_data.iloc[-1]['Close']
        sell_price = final_price * (1 - SLIPPAGE_RATE) * (1 - COMMISSION_RATE)
        cash += shares * sell_price
        trade_return = (sell_price - entry_price) / entry_price
        trades.append({
            'entry_date': entry_date,
            'exit_date': price_data.index[-1],
            'return': trade_return,
            'exit_reason': 'end'
        })

    final_value = cash
    total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL

    # 最大回撤
    portfolio_series = pd.Series(portfolio_values)
    rolling_max = portfolio_series.expanding().max()
    drawdowns = (portfolio_series - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()

    # 年化收益
    years = 5
    annualized = (1 + total_return) ** (1/years) - 1

    # 夏普率
    daily_returns_arr = np.array(daily_returns)
    if len(daily_returns_arr) > 0 and np.std(daily_returns_arr) > 0:
        sharpe = np.mean(daily_returns_arr) / np.std(daily_returns_arr) * np.sqrt(252)
    else:
        sharpe = 0

    return {
        'total_return': total_return,
        'annualized': annualized,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'trades': len(trades),
        'final_value': final_value
    }


def main():
    print("=" * 90)
    print("连续5年回测 (2021-2025) - 动态复利 vs 固定仓位")
    print("=" * 90)
    print(f"\n策略配置:")
    print(f"  买入: smoothed_index < {BUY_THRESHOLD}")
    print(f"  止损: 价格 < MA{MA_PERIOD} AND 情绪 > {STOP_SENTIMENT}")
    print(f"\n仓位计算:")
    print(f"  动态复利: 当前总资产 × {POSITION_PCT:.0%}")
    print(f"  固定仓位: 初始资金 × {POSITION_PCT:.0%} = ${INITIAL_CAPITAL * POSITION_PCT:,.0f}")

    loader = DataLoader(db_config)

    results = []

    for symbol in MAG7_SYMBOLS:
        stock_data = loader.load_ohlcv(symbol, start_date="2014-01-01")
        sentiment_data = load_sentiment_data(db_config, symbol)

        if stock_data is None or len(stock_data) == 0 or sentiment_data is None:
            continue

        # 动态复利
        dynamic = run_continuous_backtest(stock_data, sentiment_data, use_dynamic=True)
        # 固定仓位
        fixed = run_continuous_backtest(stock_data, sentiment_data, use_dynamic=False)

        if dynamic and fixed:
            results.append({
                'symbol': symbol,
                'dynamic': dynamic,
                'fixed': fixed
            })

    loader.close()

    # 打印对比结果
    print("\n" + "=" * 90)
    print("5年累计收益对比")
    print("=" * 90)

    print(f"\n{'股票':<8} {'动态复利':>15} {'固定仓位':>15} {'差异':>12} {'提升':>10}")
    print("-" * 70)

    for r in sorted(results, key=lambda x: x['dynamic']['total_return'], reverse=True):
        d = r['dynamic']['total_return']
        f = r['fixed']['total_return']
        diff = d - f
        pct = (d - f) / f * 100 if f != 0 else 0
        print(f"{r['symbol']:<8} {d:>15.2%} {f:>15.2%} {diff:>+12.2%} {pct:>+10.1f}%")

    print("-" * 70)
    avg_d = np.mean([r['dynamic']['total_return'] for r in results])
    avg_f = np.mean([r['fixed']['total_return'] for r in results])
    print(f"{'平均':<8} {avg_d:>15.2%} {avg_f:>15.2%} {avg_d - avg_f:>+12.2%}")

    # 详细指标对比
    print("\n" + "=" * 90)
    print("详细指标对比 - 动态复利")
    print("=" * 90)

    print(f"\n{'股票':<8} {'5年累计':>12} {'年化':>10} {'最大回撤':>10} {'夏普':>8} {'交易':>6}")
    print("-" * 60)

    for r in sorted(results, key=lambda x: x['dynamic']['total_return'], reverse=True):
        d = r['dynamic']
        print(f"{r['symbol']:<8} {d['total_return']:>12.2%} {d['annualized']:>10.2%} "
              f"{d['max_drawdown']:>10.2%} {d['sharpe']:>8.2f} {d['trades']:>6}")

    print("\n" + "=" * 90)
    print("详细指标对比 - 固定仓位")
    print("=" * 90)

    print(f"\n{'股票':<8} {'5年累计':>12} {'年化':>10} {'最大回撤':>10} {'夏普':>8} {'交易':>6}")
    print("-" * 60)

    for r in sorted(results, key=lambda x: x['fixed']['total_return'], reverse=True):
        f = r['fixed']
        print(f"{r['symbol']:<8} {f['total_return']:>12.2%} {f['annualized']:>10.2%} "
              f"{f['max_drawdown']:>10.2%} {f['sharpe']:>8.2f} {f['trades']:>6}")

    # 最终资产对比
    print("\n" + "=" * 90)
    print("最终资产对比 (初始 $100,000)")
    print("=" * 90)

    print(f"\n{'股票':<8} {'动态复利':>15} {'固定仓位':>15} {'差额':>15}")
    print("-" * 60)

    for r in sorted(results, key=lambda x: x['dynamic']['final_value'], reverse=True):
        d_val = r['dynamic']['final_value']
        f_val = r['fixed']['final_value']
        diff = d_val - f_val
        print(f"{r['symbol']:<8} ${d_val:>14,.0f} ${f_val:>14,.0f} ${diff:>+14,.0f}")

    print("\n" + "=" * 90)
    print("完成!")
    print("=" * 90)


if __name__ == "__main__":
    main()
