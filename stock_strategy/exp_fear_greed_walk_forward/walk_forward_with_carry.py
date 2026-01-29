"""
Walk-Forward with Position Carry + Relaxed Threshold
训练期末持仓状态延续到测试期 + 测试期放宽买入阈值

改进点:
1. 如果训练期结束时持仓 (train_end_hold=1)，测试期从持仓状态开始
2. 如果训练期结束时空仓 (train_end_hold=0)，测试期从空仓状态开始
3. 测试期买入阈值 = 训练期阈值 × THRESHOLD_RELAX_FACTOR (放宽)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime
from itertools import product
import matplotlib.pyplot as plt

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC']
plt.rcParams['axes.unicode_minus'] = False

from data_loader import DataLoader

# ============================================================
# 配置
# ============================================================
SYMBOL = "TSLA"

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

# 测试期阈值放宽系数
# 测试期买入阈值 = 训练期阈值 × THRESHOLD_RELAX_FACTOR
# 例如: 训练期 -20, 测试期 -20 × 0.8 = -16
THRESHOLD_RELAX_FACTOR = 0.8

# Walk-Forward 窗口
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
    """
    运行回测，支持初始持仓状态

    Parameters:
    -----------
    initial_position : int
        初始持仓股数 (0 = 空仓)
    initial_entry_price : float
        初始持仓成本价
    initial_cash : float
        初始现金 (如果为None，使用INITIAL_CAPITAL)

    Returns:
    --------
    final_value : float
        最终资金
    trades : list
        交易记录
    end_position : int
        期末持仓股数
    end_entry_price : float
        期末持仓成本价
    end_cash : float
        期末现金
    """
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

    # 如果有初始持仓，标记入场日期
    if position > 0:
        entry_date = df.index[0]  # 标记为期初持有

    for i in range(len(df)):
        current_date = df.index[i]
        current_price = df['Close'].iloc[i]
        current_sentiment = df['sentiment'].iloc[i]
        current_ma50 = df['MA50'].iloc[i]

        # 买入逻辑: 情绪低迷
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

        # 卖出逻辑
        elif position > 0:
            sell_signal = False
            exit_reason = None

            # OR卖出
            if current_sentiment > or_threshold:
                sell_signal = True
                exit_reason = 'OR'
            # AND卖出
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
                    'entry_price': entry_price,
                    'exit_price': sell_price,
                    'shares': position,
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'exit_reason': exit_reason
                })

                position = 0
                entry_price = 0

    # 计算最终价值
    final_value = cash + position * df['Close'].iloc[-1]

    return final_value, trades, position, entry_price, cash




def grid_search_with_carry(train_price, train_sentiment, test_price, test_sentiment):
    """
    网格搜索 + 持仓延续 + 阈值放宽

    改进:
    1. 训练期使用原始阈值
    2. 测试期买入阈值 = 训练期阈值 × THRESHOLD_RELAX_FACTOR
    3. 持仓状态延续
    """
    results = []

    for buy_t, and_t, or_t in product(BUY_THRESHOLDS, AND_SELL_THRESHOLDS, OR_THRESHOLDS):
        try:
            # 训练期回测 (使用原始阈值)
            train_final, train_trades, end_position, end_entry_price, end_cash = run_backtest_with_state(
                train_price, train_sentiment, buy_t, and_t, or_t,
                initial_position=0, initial_entry_price=0
            )

            train_return = (train_final / INITIAL_CAPITAL - 1) * 100
            train_end_hold = 1 if end_position > 0 else 0

            # 测试期买入阈值放宽
            # 例如: 训练期 -20 → 测试期 -20 × 0.8 = -16
            test_buy_t = buy_t * THRESHOLD_RELAX_FACTOR

            # 计算测试期初始价值 (修复：使用训练期结束时的实际现金)
            if end_position > 0:
                test_first_price = test_price['Close'].iloc[0]
                # 正确计算：使用训练期结束时的实际现金 + 持仓按测试期首日价格估值
                test_start_value = end_cash + end_position * test_first_price
            else:
                test_start_value = end_cash  # 空仓时，使用训练期结束时的现金

            # 测试期回测 - 延续训练期末状态 + 放宽买入阈值 + 使用正确的初始现金
            test_final, test_trades, _, _, _ = run_backtest_with_state(
                test_price, test_sentiment, test_buy_t, and_t, or_t,
                initial_position=end_position, initial_entry_price=end_entry_price,
                initial_cash=end_cash
            )

            # 测试期收益基于测试期初始价值
            test_return = (test_final / test_start_value - 1) * 100

            # 计算指标
            all_train_trades = train_trades
            train_wins = sum(1 for t in all_train_trades if t['profit'] > 0)
            train_win_rate = train_wins / len(all_train_trades) if all_train_trades else 0

            results.append({
                'buy_threshold': buy_t,
                'test_buy_threshold': test_buy_t,  # 新增: 记录测试期阈值
                'and_sell_threshold': and_t,
                'or_threshold': or_t,
                'train_return': train_return,
                'train_trades': len(train_trades),
                'train_win_rate': train_win_rate * 100,
                'train_end_hold': train_end_hold,
                'test_return': test_return,
                'test_trades': len(test_trades),
            })

        except Exception as e:
            pass

    return pd.DataFrame(results)


def run_single_window(window, price_data, sentiment_data):
    """运行单个窗口"""
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

    # 网格搜索
    results = grid_search_with_carry(train_price, train_sentiment, test_price, test_sentiment)

    if len(results) == 0:
        return None

    # Return-Based: 按训练期收益排序
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


def main():
    print("=" * 80)
    print(f"Walk-Forward 回测: {SYMBOL}")
    print("=" * 80)
    print(f"\n参数选择: Return-Based (训练期收益最高)")
    print(f"持仓延续: 训练期末持仓 → 测试期继续持有")
    print(f"阈值放宽: 测试期阈值 = 训练期 × {THRESHOLD_RELAX_FACTOR}")

    # 加载数据
    print("\n加载数据...")
    loader = DataLoader(db_config)
    price_data = loader.load_ohlcv(SYMBOL, start_date="2011-01-01")
    loader.close()
    sentiment_data = load_fear_greed_index(SYMBOL)

    print(f"  价格数据: {len(price_data)} 行")
    print(f"  情绪数据: {len(sentiment_data)} 行")

    # 运行所有窗口
    results = []
    for window in WINDOWS:
        print(f"\n{'='*60}")
        print(f"{window['name']}: Train {window['train'][0][:4]}-{window['train'][1][:4]} → Test {window['test'][0][:4]}")
        print("=" * 60)

        result = run_single_window(window, price_data, sentiment_data)

        if result:
            hold = "持仓" if result['train_hold'] else "空仓"
            print(f"\n  最优参数: {result['params']}")
            print(f"  训练期: {result['train_return']:+.1f}% | {result['train_trades']}次交易 | 期末{hold}")
            print(f"  测试期: {result['test_return']:+.1f}% | {result['test_trades']}次交易")

            results.append(result)

    # 汇总
    if results:
        print("\n" + "=" * 80)
        print("汇总结果")
        print("=" * 80)

        print(f"\n{'Window':<8} {'Test':<6} {'参数':<20} {'训练期':>10} {'测试期':>10}")
        print("-" * 70)

        for r in results:
            params_str = f"({r['params'][0]},{r['params'][1]},{r['params'][2]})"
            print(f"{r['window']:<8} {r['test_year']:<6} {params_str:<20} {r['train_return']:>+9.1f}% {r['test_return']:>+9.1f}%")

        # 累计收益
        cumret = 1
        for r in results:
            cumret *= (1 + r['test_return'] / 100)

        print("-" * 70)
        print(f"{'累计测试期收益':>36} {(cumret-1)*100:>+9.1f}%")

        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_dir = os.path.dirname(__file__)

        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(base_dir, f'walk_forward_carry_{SYMBOL}_{timestamp}.csv'), index=False)

        with open(os.path.join(base_dir, f'best_params_{SYMBOL}_carry.txt'), 'w') as f:
            f.write(f"实验: {SYMBOL} Walk-Forward 回测\n")
            f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("参数选择: Return-Based (训练期收益最高)\n")
            f.write(f"持仓延续 + 阈值放宽×{THRESHOLD_RELAX_FACTOR}\n\n")

            for r in results:
                hold = "持仓" if r['train_hold'] else "空仓"
                f.write(f"{r['window']} ({r['test_year']}): {r['params']}\n")
                f.write(f"  训练: {r['train_return']:+.1f}% ({r['train_trades']}次) 期末{hold}\n")
                f.write(f"  测试: {r['test_return']:+.1f}% ({r['test_trades']}次)\n\n")

            f.write(f"\n累计测试期收益: {(cumret-1)*100:+.1f}%\n")

        print(f"\n✅ 完成! 结果已保存")

        return results


if __name__ == "__main__":
    main()
