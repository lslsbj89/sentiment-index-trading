"""
放宽卖出条件测试
原条件: 情绪 > 30
新条件1: 情绪 > 20
新条件2: 情绪 > 5 AND 价格 < MA50
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


def run_backtest(stock_data, sentiment_data, exit_mode='original'):
    """
    连续5年回测 (2021-2025) - 动态复利

    exit_mode:
        'original': 情绪 > 30
        'relaxed_20': 情绪 > 20
        'and_5_ma': 情绪 > 5 AND 价格 < MA50
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

        # 卖出逻辑
        if shares > 0:
            sell_signal = False
            exit_reason = None

            if exit_mode == 'original':
                # 原条件: 情绪 > 30
                if sentiment > 30:
                    sell_signal = True
                    exit_reason = f'sentiment>30 ({sentiment:.0f})'

            elif exit_mode == 'relaxed_20':
                # 新条件1: 情绪 > 20
                if sentiment > 20:
                    sell_signal = True
                    exit_reason = f'sentiment>20 ({sentiment:.0f})'

            elif exit_mode == 'and_5_ma':
                # 新条件2: 情绪 > 5 AND 价格 < MA50
                if sentiment > 5 and price < ma:
                    sell_signal = True
                    exit_reason = f'sent>5&price<MA ({sentiment:.0f}, p<MA)'

            if sell_signal:
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
                    'exit_reason': exit_reason,
                    'proceeds': proceeds
                })
                shares = 0
                entry_price = 0

        # 买入逻辑 - 动态复利
        elif shares == 0 and sentiment < BUY_THRESHOLD:
            current_total = cash
            position_value = current_total * POSITION_PCT

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

    # 胜率
    if len(trades) > 0:
        wins = sum(1 for t in trades if t['return'] > 0)
        win_rate = wins / len(trades)
    else:
        win_rate = 0

    return {
        'total_return': total_return,
        'annualized': annualized,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'trades': len(trades),
        'win_rate': win_rate,
        'final_value': final_value,
        'trade_details': trades
    }


def main():
    print("=" * 100)
    print("放宽卖出条件测试 - 动态复利仓位")
    print("=" * 100)
    print(f"\n买入条件: smoothed_index < {BUY_THRESHOLD}")
    print(f"\n卖出条件对比:")
    print(f"  原条件:     情绪 > 30")
    print(f"  新条件1:    情绪 > 20")
    print(f"  新条件2:    情绪 > 5 AND 价格 < MA{MA_PERIOD}")

    loader = DataLoader(db_config)

    results = []

    for symbol in MAG7_SYMBOLS:
        stock_data = loader.load_ohlcv(symbol, start_date="2014-01-01")
        sentiment_data = load_sentiment_data(db_config, symbol)

        if stock_data is None or len(stock_data) == 0 or sentiment_data is None:
            continue

        # 三种卖出条件
        original = run_backtest(stock_data, sentiment_data, exit_mode='original')
        relaxed_20 = run_backtest(stock_data, sentiment_data, exit_mode='relaxed_20')
        and_5_ma = run_backtest(stock_data, sentiment_data, exit_mode='and_5_ma')

        if original and relaxed_20 and and_5_ma:
            results.append({
                'symbol': symbol,
                'original': original,
                'relaxed_20': relaxed_20,
                'and_5_ma': and_5_ma
            })

    loader.close()

    # 打印5年累计收益对比
    print("\n" + "=" * 100)
    print("5年累计收益对比 (2021-2025)")
    print("=" * 100)

    print(f"\n{'股票':<8} {'原(>30)':>12} {'新1(>20)':>12} {'新2(>5&MA)':>14} {'最优':>12}")
    print("-" * 70)

    for r in sorted(results, key=lambda x: max(x['original']['total_return'],
                                                x['relaxed_20']['total_return'],
                                                x['and_5_ma']['total_return']), reverse=True):
        o = r['original']['total_return']
        r20 = r['relaxed_20']['total_return']
        and5 = r['and_5_ma']['total_return']

        best_val = max(o, r20, and5)
        if best_val == o:
            best = "原(>30)"
        elif best_val == r20:
            best = "新1(>20)"
        else:
            best = "新2(>5&MA)"

        print(f"{r['symbol']:<8} {o:>12.2%} {r20:>12.2%} {and5:>14.2%} {best:>12}")

    print("-" * 70)
    avg_o = np.mean([r['original']['total_return'] for r in results])
    avg_r20 = np.mean([r['relaxed_20']['total_return'] for r in results])
    avg_and5 = np.mean([r['and_5_ma']['total_return'] for r in results])
    print(f"{'平均':<8} {avg_o:>12.2%} {avg_r20:>12.2%} {avg_and5:>14.2%}")

    # 打印交易次数对比
    print("\n" + "=" * 100)
    print("交易次数对比")
    print("=" * 100)

    print(f"\n{'股票':<8} {'原(>30)':>12} {'新1(>20)':>12} {'新2(>5&MA)':>14}")
    print("-" * 55)

    for r in results:
        o = r['original']['trades']
        r20 = r['relaxed_20']['trades']
        and5 = r['and_5_ma']['trades']
        print(f"{r['symbol']:<8} {o:>12} {r20:>12} {and5:>14}")

    print("-" * 55)
    total_o = sum([r['original']['trades'] for r in results])
    total_r20 = sum([r['relaxed_20']['trades'] for r in results])
    total_and5 = sum([r['and_5_ma']['trades'] for r in results])
    print(f"{'总计':<8} {total_o:>12} {total_r20:>12} {total_and5:>14}")

    # 打印详细指标
    print("\n" + "=" * 100)
    print("详细指标对比")
    print("=" * 100)

    for r in results:
        print(f"\n--- {r['symbol']} ---")
        print(f"  {'指标':<12} {'原(>30)':>12} {'新1(>20)':>12} {'新2(>5&MA)':>14}")
        print(f"  {'-'*55}")

        o = r['original']
        r20 = r['relaxed_20']
        and5 = r['and_5_ma']

        print(f"  {'5年累计':<12} {o['total_return']:>12.2%} {r20['total_return']:>12.2%} {and5['total_return']:>14.2%}")
        print(f"  {'年化收益':<12} {o['annualized']:>12.2%} {r20['annualized']:>12.2%} {and5['annualized']:>14.2%}")
        print(f"  {'最大回撤':<12} {o['max_drawdown']:>12.2%} {r20['max_drawdown']:>12.2%} {and5['max_drawdown']:>14.2%}")
        print(f"  {'夏普率':<12} {o['sharpe']:>12.2f} {r20['sharpe']:>12.2f} {and5['sharpe']:>14.2f}")
        print(f"  {'交易次数':<12} {o['trades']:>12} {r20['trades']:>12} {and5['trades']:>14}")
        print(f"  {'胜率':<12} {o['win_rate']:>12.2%} {r20['win_rate']:>12.2%} {and5['win_rate']:>14.2%}")

    # 打印交易详情
    print("\n" + "=" * 100)
    print("交易详情 (新条件2: >5 AND <MA)")
    print("=" * 100)

    for r in results:
        trades = r['and_5_ma']['trade_details']
        if len(trades) > 1:  # 只显示有多笔交易的
            print(f"\n--- {r['symbol']} ({len(trades)} 笔交易) ---")
            for i, t in enumerate(trades[:10], 1):  # 最多显示10笔
                entry = t['entry_date'].strftime('%Y-%m-%d') if hasattr(t['entry_date'], 'strftime') else str(t['entry_date'])[:10]
                exit_d = t['exit_date'].strftime('%Y-%m-%d') if hasattr(t['exit_date'], 'strftime') else str(t['exit_date'])[:10]
                print(f"  {i}. {entry} → {exit_d}: {t['return']:+.2%} ({t['exit_reason']})")

    # 最终建议
    print("\n" + "=" * 100)
    print("结论")
    print("=" * 100)

    best_strategy = None
    best_avg = max(avg_o, avg_r20, avg_and5)
    if best_avg == avg_o:
        best_strategy = "原条件 (>30)"
    elif best_avg == avg_r20:
        best_strategy = "新条件1 (>20)"
    else:
        best_strategy = "新条件2 (>5 AND <MA)"

    print(f"\n平均收益最高: {best_strategy} ({best_avg:.2%})")

    # 按股票推荐
    print("\n各股票最优策略:")
    for r in results:
        o = r['original']['total_return']
        r20 = r['relaxed_20']['total_return']
        and5 = r['and_5_ma']['total_return']

        best_val = max(o, r20, and5)
        if best_val == o:
            best = "原(>30)"
        elif best_val == r20:
            best = "新1(>20)"
        else:
            best = "新2(>5&MA)"

        diff_from_original = best_val - o
        print(f"  {r['symbol']:<8}: {best:<12} ({best_val:+.2%}, 比原条件{diff_from_original:+.2%})")

    print("\n" + "=" * 100)
    print("完成!")
    print("=" * 100)


if __name__ == "__main__":
    main()
