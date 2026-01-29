"""
最优策略配置验证
按照每只股票的最优配置运行5年连续回测
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import psycopg2

from data_loader import DataLoader

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

# 最优配置表
OPTIMAL_CONFIG = {
    'NVDA': {'sell_mode': 'and_10_ma', 'position_mode': 'dynamic'},
    'TSLA': {'sell_mode': 'and_10_ma', 'position_mode': 'dynamic'},
    'META': {'sell_mode': 'and_5_ma', 'position_mode': 'dynamic'},
    'MSFT': {'sell_mode': 'original_30', 'position_mode': 'fixed'},
    'GOOGL': {'sell_mode': 'original_30', 'position_mode': 'fixed'},
    'AAPL': {'sell_mode': 'original_30', 'position_mode': 'fixed'},
    'AMZN': {'sell_mode': 'and_5_ma', 'position_mode': 'fixed'},
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


def run_optimal_backtest(stock_data, sentiment_data, symbol):
    """
    按照最优配置运行5年连续回测
    """
    config = OPTIMAL_CONFIG[symbol]
    sell_mode = config['sell_mode']
    use_dynamic = config['position_mode'] == 'dynamic'

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
    fixed_position_value = INITIAL_CAPITAL * POSITION_PCT

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

        # 卖出逻辑 - 根据配置
        if shares > 0:
            sell_signal = False
            exit_reason = None

            if sell_mode == 'original_30':
                if sentiment > 30:
                    sell_signal = True
                    exit_reason = f'sentiment>30 ({sentiment:.0f})'

            elif sell_mode == 'and_10_ma':
                if sentiment > 10 and price < ma:
                    sell_signal = True
                    exit_reason = f'>10&<MA ({sentiment:.0f})'

            elif sell_mode == 'and_5_ma':
                if sentiment > 5 and price < ma:
                    sell_signal = True
                    exit_reason = f'>5&<MA ({sentiment:.0f})'

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
                    'exit_reason': exit_reason
                })
                shares = 0
                entry_price = 0

        # 买入逻辑
        elif shares == 0 and sentiment < BUY_THRESHOLD:
            if use_dynamic:
                position_value = cash * POSITION_PCT
            else:
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

    # 胜率
    if len(trades) > 0:
        wins = sum(1 for t in trades if t['return'] > 0)
        win_rate = wins / len(trades)
    else:
        win_rate = 0

    return {
        'symbol': symbol,
        'sell_mode': sell_mode,
        'position_mode': config['position_mode'],
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
    print("最优策略配置验证 - 连续5年回测 (2021-2025)")
    print("=" * 100)

    print("\n最优配置表:")
    print("-" * 70)
    print(f"{'股票':<8} {'卖出条件':<20} {'仓位管理':<12}")
    print("-" * 70)
    for symbol, config in OPTIMAL_CONFIG.items():
        sell_desc = {
            'original_30': '情绪 > 30',
            'and_10_ma': '情绪 > 10 AND 价格 < MA50',
            'and_5_ma': '情绪 > 5 AND 价格 < MA50'
        }[config['sell_mode']]
        pos_desc = '动态复利' if config['position_mode'] == 'dynamic' else '固定仓位'
        print(f"{symbol:<8} {sell_desc:<20} {pos_desc:<12}")

    loader = DataLoader(db_config)

    results = []

    for symbol in OPTIMAL_CONFIG.keys():
        stock_data = loader.load_ohlcv(symbol, start_date="2014-01-01")
        sentiment_data = load_sentiment_data(db_config, symbol)

        if stock_data is None or len(stock_data) == 0 or sentiment_data is None:
            continue

        result = run_optimal_backtest(stock_data, sentiment_data, symbol)
        if result:
            results.append(result)

    loader.close()

    # 打印验证结果
    print("\n" + "=" * 100)
    print("验证结果")
    print("=" * 100)

    print(f"\n{'股票':<8} {'卖出条件':<15} {'仓位':<8} {'5年收益':>12} {'年化':>10} {'最大回撤':>10} {'夏普':>8} {'交易':>6} {'胜率':>8}")
    print("-" * 95)

    for r in sorted(results, key=lambda x: x['total_return'], reverse=True):
        sell_short = {
            'original_30': '>30',
            'and_10_ma': '>10&MA',
            'and_5_ma': '>5&MA'
        }[r['sell_mode']]
        pos_short = '动态' if r['position_mode'] == 'dynamic' else '固定'

        print(f"{r['symbol']:<8} {sell_short:<15} {pos_short:<8} {r['total_return']:>12.2%} "
              f"{r['annualized']:>10.2%} {r['max_drawdown']:>10.2%} {r['sharpe']:>8.2f} "
              f"{r['trades']:>6} {r['win_rate']:>8.0%}")

    print("-" * 95)

    # 汇总统计
    avg_return = np.mean([r['total_return'] for r in results])
    avg_annual = np.mean([r['annualized'] for r in results])
    avg_dd = np.mean([r['max_drawdown'] for r in results])
    avg_sharpe = np.mean([r['sharpe'] for r in results])
    total_trades = sum([r['trades'] for r in results])

    print(f"{'平均':<8} {'':<15} {'':<8} {avg_return:>12.2%} {avg_annual:>10.2%} "
          f"{avg_dd:>10.2%} {avg_sharpe:>8.2f} {total_trades:>6}")

    # 打印交易详情
    print("\n" + "=" * 100)
    print("交易详情")
    print("=" * 100)

    for r in results:
        if len(r['trade_details']) > 0:
            print(f"\n--- {r['symbol']} ({len(r['trade_details'])} 笔交易) ---")
            for i, t in enumerate(r['trade_details'], 1):
                entry = t['entry_date'].strftime('%Y-%m-%d') if hasattr(t['entry_date'], 'strftime') else str(t['entry_date'])[:10]
                exit_d = t['exit_date'].strftime('%Y-%m-%d') if hasattr(t['exit_date'], 'strftime') else str(t['exit_date'])[:10]
                print(f"  {i}. {entry} → {exit_d}: {t['return']:+.2%} ({t['exit_reason']})")

    # 最终资产
    print("\n" + "=" * 100)
    print("最终资产 (初始 $100,000)")
    print("=" * 100)

    print(f"\n{'股票':<8} {'最终资产':>15} {'净利润':>15}")
    print("-" * 45)

    total_final = 0
    for r in sorted(results, key=lambda x: x['final_value'], reverse=True):
        profit = r['final_value'] - INITIAL_CAPITAL
        total_final += r['final_value']
        print(f"{r['symbol']:<8} ${r['final_value']:>14,.0f} ${profit:>+14,.0f}")

    print("-" * 45)
    total_invested = INITIAL_CAPITAL * len(results)
    total_profit = total_final - total_invested
    print(f"{'合计':<8} ${total_final:>14,.0f} ${total_profit:>+14,.0f}")
    print(f"{'平均':<8} ${total_final/len(results):>14,.0f} ${total_profit/len(results):>+14,.0f}")

    print("\n" + "=" * 100)
    print("验证完成!")
    print("=" * 100)


if __name__ == "__main__":
    main()
