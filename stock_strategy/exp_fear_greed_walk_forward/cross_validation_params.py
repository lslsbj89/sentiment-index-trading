"""
交叉验证 - 用固定参数测试所有窗口
比较不同候选参数组合的综合表现
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime
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

# 回测参数
INITIAL_CAPITAL = 100000
COMMISSION_RATE = 0.001
SLIPPAGE_RATE = 0.001
POSITION_PCT = 0.8

# Walk-Forward 窗口
WINDOWS = [
    {"name": "W2020", "train": ("2016-01-01", "2019-12-31"), "test": ("2020-01-01", "2020-12-31")},
    {"name": "W2021", "train": ("2017-01-01", "2020-12-31"), "test": ("2021-01-01", "2021-12-31")},
    {"name": "W2022", "train": ("2018-01-01", "2021-12-31"), "test": ("2022-01-01", "2022-12-31")},
    {"name": "W2023", "train": ("2019-01-01", "2022-12-31"), "test": ("2023-01-01", "2023-12-31")},
    {"name": "W2024", "train": ("2020-01-01", "2023-12-31"), "test": ("2024-01-01", "2024-12-31")},
    {"name": "W2025", "train": ("2021-01-01", "2024-12-31"), "test": ("2025-01-01", "2025-12-31")},
]

# 候选通用参数
CANDIDATES = [
    {"name": "简单平均", "buy": -15, "and_sell": 15, "or_sell": 55},
    {"name": "加权平均", "buy": -15, "and_sell": 20, "or_sell": 50},
    {"name": "众数", "buy": -20, "and_sell": 10, "or_sell": 45},
    {"name": "中位数", "buy": -18, "and_sell": 12, "or_sell": 55},
    {"name": "保守折中", "buy": -15, "and_sell": 15, "or_sell": 45},
    {"name": "宽松买入", "buy": -10, "and_sell": 15, "or_sell": 50},
    {"name": "严格买入", "buy": -25, "and_sell": 15, "or_sell": 50},
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


def run_backtest(price_data, sentiment_data, buy_threshold, and_sell_threshold, or_threshold):
    """运行回测，返回详细结果"""
    df = price_data.copy()
    df['sentiment'] = sentiment_data['smoothed_index'].reindex(df.index)
    df['MA50'] = df['Close'].rolling(50).mean()
    df = df.dropna()

    if len(df) < 10:
        return {
            'final_value': INITIAL_CAPITAL,
            'return_pct': 0,
            'trades': 0,
            'wins': 0,
            'max_drawdown': 0
        }

    cash = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    trades = []

    # 跟踪最大回撤
    peak_value = INITIAL_CAPITAL
    max_drawdown = 0
    portfolio_values = []

    for i in range(len(df)):
        current_price = df['Close'].iloc[i]
        current_sentiment = df['sentiment'].iloc[i]
        current_ma50 = df['MA50'].iloc[i]

        # 计算当前组合价值
        current_value = cash + position * current_price
        portfolio_values.append(current_value)

        # 更新最大回撤
        if current_value > peak_value:
            peak_value = current_value
        drawdown = (peak_value - current_value) / peak_value * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown

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

        # 卖出
        elif position > 0:
            sell_signal = False

            if current_sentiment > or_threshold:
                sell_signal = True
            elif current_sentiment > and_sell_threshold and current_price < current_ma50:
                sell_signal = True

            if sell_signal:
                sell_price = current_price * (1 - SLIPPAGE_RATE) * (1 - COMMISSION_RATE)
                revenue = position * sell_price
                cash += revenue

                profit_pct = (sell_price - entry_price) / entry_price * 100
                trades.append(profit_pct)

                position = 0
                entry_price = 0

    final_value = cash + position * df['Close'].iloc[-1]
    return_pct = (final_value / INITIAL_CAPITAL - 1) * 100
    wins = sum(1 for t in trades if t > 0)

    return {
        'final_value': final_value,
        'return_pct': return_pct,
        'trades': len(trades),
        'wins': wins,
        'win_rate': wins / len(trades) * 100 if trades else 0,
        'max_drawdown': max_drawdown
    }


def test_candidate(candidate, price_data, sentiment_data):
    """测试一组候选参数在所有窗口的表现"""
    results = []

    for window in WINDOWS:
        test_start = pd.Timestamp(window['test'][0], tz='UTC')
        test_end = pd.Timestamp(window['test'][1], tz='UTC')

        test_price = price_data[(price_data.index >= test_start) & (price_data.index <= test_end)]
        test_sentiment = sentiment_data[(sentiment_data.index >= test_start) & (sentiment_data.index <= test_end)]

        if len(test_price) < 50:
            continue

        result = run_backtest(
            test_price, test_sentiment,
            candidate['buy'], candidate['and_sell'], candidate['or_sell']
        )

        results.append({
            'window': window['name'],
            'return_pct': result['return_pct'],
            'trades': result['trades'],
            'win_rate': result['win_rate'],
            'max_drawdown': result['max_drawdown']
        })

    return results


def main():
    print("=" * 70)
    print(f"交叉验证 - {SYMBOL} 固定参数测试")
    print("=" * 70)

    # 加载数据
    loader = DataLoader(db_config)
    price_data = loader.load_ohlcv(SYMBOL, start_date="2015-01-01")
    loader.close()
    sentiment_data = load_fear_greed_index(SYMBOL)

    all_results = []

    # 测试每组候选参数
    for candidate in CANDIDATES:
        print(f"\n测试: {candidate['name']} (Buy<{candidate['buy']}, AND>{candidate['and_sell']}, OR>{candidate['or_sell']})")

        results = test_candidate(candidate, price_data, sentiment_data)

        # 计算综合指标
        returns = [r['return_pct'] for r in results]
        cumret = 1
        for r in returns:
            cumret *= (1 + r / 100)
        cumulative = (cumret - 1) * 100

        avg_return = np.mean(returns)
        win_windows = sum(1 for r in returns if r > 0)
        total_trades = sum(r['trades'] for r in results)
        avg_drawdown = np.mean([r['max_drawdown'] for r in results])

        print(f"  各窗口收益: {[f'{r:.1f}%' for r in returns]}")
        print(f"  累计复利: {cumulative:+.1f}%")
        print(f"  盈利窗口: {win_windows}/6")
        print(f"  总交易数: {total_trades}")

        all_results.append({
            'name': candidate['name'],
            'buy': candidate['buy'],
            'and_sell': candidate['and_sell'],
            'or_sell': candidate['or_sell'],
            'cumulative': cumulative,
            'avg_return': avg_return,
            'win_windows': win_windows,
            'total_trades': total_trades,
            'avg_drawdown': avg_drawdown,
            'returns': returns
        })

    # 排序并显示结果
    print("\n" + "=" * 70)
    print("综合排名 (按累计收益)")
    print("=" * 70)

    all_results.sort(key=lambda x: x['cumulative'], reverse=True)

    print(f"\n{'排名':<4} {'参数组合':<12} {'参数':<20} {'累计收益':>10} {'盈利窗口':>10} {'交易数':>8}")
    print("-" * 70)

    for i, r in enumerate(all_results):
        params = f"({r['buy']},{r['and_sell']},{r['or_sell']})"
        print(f"{i+1:<4} {r['name']:<12} {params:<20} {r['cumulative']:>+10.1f}% {r['win_windows']:>10}/6 {r['total_trades']:>8}")

    # 详细对比表
    print("\n" + "=" * 70)
    print("各窗口收益详细对比")
    print("=" * 70)

    header = f"{'参数组合':<12}"
    for w in WINDOWS:
        header += f" {w['name']:>8}"
    header += f" {'累计':>10}"
    print(header)
    print("-" * 70)

    for r in all_results:
        row = f"{r['name']:<12}"
        for ret in r['returns']:
            row += f" {ret:>+8.1f}%"
        row += f" {r['cumulative']:>+10.1f}%"
        print(row)

    # 推荐
    print("\n" + "=" * 70)
    print("推荐")
    print("=" * 70)

    best = all_results[0]
    print(f"\n最佳参数: {best['name']}")
    print(f"  Buy < {best['buy']}, AND > {best['and_sell']}, OR > {best['or_sell']}")
    print(f"  累计收益: {best['cumulative']:+.1f}%")
    print(f"  盈利窗口: {best['win_windows']}/6")

    # 稳定性最好的 (盈利窗口最多)
    most_stable = max(all_results, key=lambda x: (x['win_windows'], x['cumulative']))
    if most_stable['name'] != best['name']:
        print(f"\n最稳定参数: {most_stable['name']}")
        print(f"  Buy < {most_stable['buy']}, AND > {most_stable['and_sell']}, OR > {most_stable['or_sell']}")
        print(f"  累计收益: {most_stable['cumulative']:+.1f}%")
        print(f"  盈利窗口: {most_stable['win_windows']}/6")

    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df = pd.DataFrame(all_results)
    df.to_csv(f'cross_validation_{SYMBOL}_{timestamp}.csv', index=False)
    print(f"\n结果已保存: cross_validation_{SYMBOL}_{timestamp}.csv")


if __name__ == "__main__":
    main()
