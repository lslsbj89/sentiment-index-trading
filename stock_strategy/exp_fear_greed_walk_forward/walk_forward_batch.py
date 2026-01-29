"""
Walk-Forward 批量测试 - 美股七姐妹
TSLA, AAPL, MSFT, GOOGL, AMZN, NVDA, META
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

# ============================================================
# 配置
# ============================================================
SYMBOLS = ["TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"]

db_config = {
    "host": "localhost",
    "port": 5432,
    "database": "crypto_fear_greed_2",
    "user": "sc2025",
    "password": ""
}

# 网格搜索参数
BUY_THRESHOLDS = [-30, -25, -20, -15, -10, -5, 0, 5]
AND_SELL_THRESHOLDS = [5, 10, 15, 20, 25]
OR_THRESHOLDS = [25, 30, 35, 40, 45, 50, 55, 60, 65]

# 回测参数
INITIAL_CAPITAL = 100000
COMMISSION_RATE = 0.001
SLIPPAGE_RATE = 0.001
POSITION_PCT = 0.8

# 阈值放宽系数
THRESHOLD_RELAX_FACTOR = 0.8

# Walk-Forward 窗口 (2020-2025)
WINDOWS = [
    {"name": "W2020", "train": ("2016-01-01", "2019-12-31"), "test": ("2020-01-01", "2020-12-31")},
    {"name": "W2021", "train": ("2017-01-01", "2020-12-31"), "test": ("2021-01-01", "2021-12-31")},
    {"name": "W2022", "train": ("2018-01-01", "2021-12-31"), "test": ("2022-01-01", "2022-12-31")},
    {"name": "W2023", "train": ("2019-01-01", "2022-12-31"), "test": ("2023-01-01", "2023-12-31")},
    {"name": "W2024", "train": ("2020-01-01", "2023-12-31"), "test": ("2024-01-01", "2024-12-31")},
    {"name": "W2025", "train": ("2021-01-01", "2024-12-31"), "test": ("2025-01-01", "2025-12-31")},
]


def load_fear_greed_index(symbol):
    conn = psycopg2.connect(**db_config)
    query = """
        SELECT date, smoothed_index
        FROM fear_greed_index
        WHERE symbol = %s
        ORDER BY date
    """
    df = pd.read_sql(query, conn, params=(symbol,))
    conn.close()
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.set_index('date')
    return df


def run_backtest_with_state(price_data, sentiment_data, buy_threshold, and_sell_threshold,
                            or_threshold, initial_position=0, initial_entry_price=0, initial_cash=None):
    df = price_data.copy()
    df['sentiment'] = sentiment_data['smoothed_index'].reindex(df.index)
    df['MA50'] = df['Close'].rolling(50).mean()
    df = df.dropna()

    if len(df) < 10:
        return INITIAL_CAPITAL, [], 0, 0, INITIAL_CAPITAL

    # 使用传入的初始现金，如果没有则使用INITIAL_CAPITAL
    if initial_cash is not None:
        cash = initial_cash
    else:
        cash = INITIAL_CAPITAL

    position = initial_position
    entry_price = initial_entry_price
    entry_date = None
    trades = []

    if position > 0:
        entry_date = df.index[0]

    for i in range(len(df)):
        current_date = df.index[i]
        current_price = df['Close'].iloc[i]
        current_sentiment = df['sentiment'].iloc[i]
        current_ma50 = df['MA50'].iloc[i]

        # 买入
        if position == 0 and current_sentiment < buy_threshold:
            available = cash * POSITION_PCT
            buy_price = current_price * (1 + SLIPPAGE_RATE) * (1 + COMMISSION_RATE)
            shares = int(available / buy_price)

            if shares > 0:
                cost = shares * buy_price
                cash -= cost
                position = shares
                entry_price = buy_price
                entry_date = current_date

        # 卖出
        elif position > 0:
            sell_signal = False
            exit_reason = None

            if current_sentiment > or_threshold:
                sell_signal = True
                exit_reason = 'OR'
            elif current_sentiment > and_sell_threshold and current_price < current_ma50:
                sell_signal = True
                exit_reason = 'AND'

            if sell_signal:
                sell_price = current_price * (1 - SLIPPAGE_RATE) * (1 - COMMISSION_RATE)
                revenue = position * sell_price
                cash += revenue

                profit = revenue - position * entry_price
                profit_pct = profit / (position * entry_price)

                trades.append({
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'profit_pct': profit_pct,
                    'exit_reason': exit_reason
                })

                position = 0
                entry_price = 0

    final_value = cash + position * df['Close'].iloc[-1]
    return final_value, trades, position, entry_price, cash


def grid_search_with_carry(train_price, train_sentiment, test_price, test_sentiment):
    results = []

    for buy_t, and_t, or_t in product(BUY_THRESHOLDS, AND_SELL_THRESHOLDS, OR_THRESHOLDS):
        try:
            # 训练期
            train_final, train_trades, end_position, end_entry_price, end_cash = run_backtest_with_state(
                train_price, train_sentiment, buy_t, and_t, or_t,
                initial_position=0, initial_entry_price=0
            )

            train_return = (train_final / INITIAL_CAPITAL - 1) * 100
            train_end_hold = 1 if end_position > 0 else 0

            # 测试期 (阈值放宽 + 持仓延续)
            test_buy_t = buy_t * THRESHOLD_RELAX_FACTOR

            # 计算测试期初始价值 (修复：使用训练期结束时的实际现金)
            if end_position > 0:
                test_first_price = test_price['Close'].iloc[0]
                test_start_value = end_cash + end_position * test_first_price
            else:
                test_start_value = end_cash

            test_final, test_trades, _, _, _ = run_backtest_with_state(
                test_price, test_sentiment, test_buy_t, and_t, or_t,
                initial_position=end_position, initial_entry_price=end_entry_price,
                initial_cash=end_cash
            )

            # 测试期收益基于测试期初始价值
            test_return = (test_final / test_start_value - 1) * 100

            results.append({
                'buy_threshold': buy_t,
                'test_buy_threshold': test_buy_t,
                'and_sell_threshold': and_t,
                'or_threshold': or_t,
                'train_return': train_return,
                'train_trades': len(train_trades),
                'train_end_hold': train_end_hold,
                'test_return': test_return,
                'test_trades': len(test_trades),
            })

        except Exception as e:
            pass

    return pd.DataFrame(results)


def run_single_window(window, price_data, sentiment_data):
    train_start = pd.Timestamp(window['train'][0], tz='UTC')
    train_end = pd.Timestamp(window['train'][1], tz='UTC')
    test_start = pd.Timestamp(window['test'][0], tz='UTC')
    test_end = pd.Timestamp(window['test'][1], tz='UTC')

    train_price = price_data[(price_data.index >= train_start) & (price_data.index <= train_end)]
    test_price = price_data[(price_data.index >= test_start) & (price_data.index <= test_end)]
    train_sentiment = sentiment_data[(sentiment_data.index >= train_start) & (sentiment_data.index <= train_end)]
    test_sentiment = sentiment_data[(sentiment_data.index >= test_start) & (sentiment_data.index <= test_end)]

    if len(train_price) < 100 or len(test_price) < 50:
        return None

    results = grid_search_with_carry(train_price, train_sentiment, test_price, test_sentiment)

    if len(results) == 0:
        return None

    # Return-Based
    by_return = results.sort_values('train_return', ascending=False).iloc[0]

    return {
        'window': window['name'],
        'train_period': f"{window['train'][0][:4]}-{window['train'][1][:4]}",
        'test_year': window['test'][0][:4],
        'params': (int(by_return['buy_threshold']), int(by_return['and_sell_threshold']), int(by_return['or_threshold'])),
        'test_buy': by_return['test_buy_threshold'],
        'train_return': by_return['train_return'],
        'train_trades': by_return['train_trades'],
        'train_hold': by_return['train_end_hold'],
        'test_return': by_return['test_return'],
        'test_trades': by_return['test_trades'],
    }


def run_symbol(symbol):
    """运行单个股票的回测"""
    print(f"\n{'='*80}")
    print(f"  {symbol}")
    print(f"{'='*80}")

    # 加载数据
    loader = DataLoader(db_config)
    price_data = loader.load_ohlcv(symbol, start_date="2011-01-01")
    loader.close()
    sentiment_data = load_fear_greed_index(symbol)

    if len(price_data) < 500 or len(sentiment_data) < 500:
        print(f"  数据不足，跳过")
        return None

    # 运行所有窗口
    results = []
    for window in WINDOWS:
        result = run_single_window(window, price_data, sentiment_data)
        if result:
            results.append(result)

    if not results:
        print(f"  无有效结果")
        return None

    # 打印结果
    print(f"\n  {'Window':<8} {'参数':<15} {'训练期':>10} {'测试期':>10}")
    print(f"  {'-'*50}")

    for r in results:
        params_str = f"({r['params'][0]},{r['params'][1]},{r['params'][2]})"
        print(f"  {r['window']:<8} {params_str:<15} {r['train_return']:>+9.1f}% {r['test_return']:>+9.1f}%")

    # 累计收益
    cumret = 1
    for r in results:
        cumret *= (1 + r['test_return'] / 100)

    print(f"  {'-'*50}")
    print(f"  {'累计测试期收益':>25} {(cumret-1)*100:>+9.1f}%")

    return {
        'symbol': symbol,
        'results': results,
        'cumulative_return': (cumret - 1) * 100
    }


def main():
    print("=" * 80)
    print("Walk-Forward 批量测试 - 美股七姐妹")
    print("=" * 80)
    print(f"\n测试股票: {', '.join(SYMBOLS)}")
    print(f"测试窗口: 2020-2025 (6个窗口)")
    print(f"参数选择: Return-Based")
    print(f"阈值放宽: ×{THRESHOLD_RELAX_FACTOR}")

    all_results = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.dirname(__file__)

    for symbol in SYMBOLS:
        result = run_symbol(symbol)
        if result:
            all_results.append(result)

            # 保存每个股票的详细结果
            df = pd.DataFrame(result['results'])
            df.to_csv(os.path.join(base_dir, f'walk_forward_{symbol}_{timestamp}.csv'), index=False)

    # 汇总对比
    print("\n" + "=" * 80)
    print("汇总对比 - 累计测试期收益 (2020-2025)")
    print("=" * 80)

    summary = []
    for r in sorted(all_results, key=lambda x: x['cumulative_return'], reverse=True):
        print(f"  {r['symbol']:<6} {r['cumulative_return']:>+10.1f}%")
        summary.append({
            'symbol': r['symbol'],
            'cumulative_return': r['cumulative_return'],
            'num_windows': len(r['results'])
        })

    # 保存汇总
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(base_dir, f'walk_forward_summary_{timestamp}.csv'), index=False)

    # 保存详细报告
    with open(os.path.join(base_dir, f'walk_forward_report_{timestamp}.txt'), 'w') as f:
        f.write("Walk-Forward 批量测试报告 - 美股七姐妹\n")
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        f.write("测试配置:\n")
        f.write(f"  股票: {', '.join(SYMBOLS)}\n")
        f.write(f"  窗口: 2020-2025 (4年训练 + 1年测试)\n")
        f.write(f"  参数选择: Return-Based (训练期收益最高)\n")
        f.write(f"  阈值放宽: ×{THRESHOLD_RELAX_FACTOR}\n")
        f.write(f"  持仓延续: 训练期末持仓延续到测试期\n\n")

        f.write("=" * 60 + "\n")
        f.write("累计测试期收益排名 (2020-2025)\n")
        f.write("=" * 60 + "\n\n")

        for i, r in enumerate(sorted(all_results, key=lambda x: x['cumulative_return'], reverse=True), 1):
            f.write(f"  {i}. {r['symbol']:<6} {r['cumulative_return']:>+10.1f}%\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("各股票详细结果\n")
        f.write("=" * 60 + "\n\n")

        for r in all_results:
            f.write(f"\n{r['symbol']}:\n")
            f.write("-" * 40 + "\n")
            for w in r['results']:
                hold = "持仓" if w['train_hold'] else "空仓"
                f.write(f"  {w['window']} ({w['test_year']}): {w['params']}\n")
                f.write(f"    训练: {w['train_return']:+.1f}% ({w['train_trades']}次) 期末{hold}\n")
                f.write(f"    测试: {w['test_return']:+.1f}% ({w['test_trades']}次)\n")
            f.write(f"\n  累计: {r['cumulative_return']:+.1f}%\n")

    print(f"\n✅ 完成! 结果已保存到:")
    print(f"   walk_forward_summary_{timestamp}.csv")
    print(f"   walk_forward_report_{timestamp}.txt")

    return all_results


if __name__ == "__main__":
    main()
