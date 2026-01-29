"""
获取2026年卖出参数 + 历史所有窗口参数
=========================================
W2026训练期: 2022-01-01 ~ 2025-12-31
网格搜索: AND × OR = 25组合
训练期使用原版阈值分批买入
"""

import sys
sys.path.insert(0, '/Users/sc2025/Desktop/Claude/Quantrade/src')

import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime
from data_loader import DataLoader

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "crypto_fear_greed_2",
    "user": "sc2025",
    "password": ""
}

INITIAL_CAPITAL = 100000
COMMISSION = 0.001
SLIPPAGE = 0.001

BUY_THRESHOLDS = [5, 0, -5, -10]
BATCH_PCT = 0.25

AND_SELL_RANGE = [5, 10, 15, 20, 25]
OR_SELL_RANGE = [30, 40, 50, 55, 60]

WINDOWS = [
    {"name": "W2020", "train": ("2016-01-01", "2019-12-31")},
    {"name": "W2021", "train": ("2017-01-01", "2020-12-31")},
    {"name": "W2022", "train": ("2018-01-01", "2021-12-31")},
    {"name": "W2023", "train": ("2019-01-01", "2022-12-31")},
    {"name": "W2024", "train": ("2020-01-01", "2023-12-31")},
    {"name": "W2025", "train": ("2021-01-01", "2024-12-31")},
    {"name": "W2026", "train": ("2022-01-01", "2025-12-31")},
]

SYMBOLS = ["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META"]


def load_sentiment_s3(symbol):
    conn = psycopg2.connect(**DB_CONFIG)
    query = """
        SELECT date, smoothed_index
        FROM fear_greed_index_s3
        WHERE symbol = %s
        ORDER BY date
    """
    df = pd.read_sql(query, conn, params=(symbol,))
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.set_index('date')
    conn.close()
    return df


def load_price(symbol):
    loader = DataLoader(DB_CONFIG)
    price_df = loader.load_ohlcv(symbol, start_date="2011-01-01")
    loader.close()
    return price_df


def prepare_data(price_df, sentiment_df, start_date, end_date):
    df = price_df.copy()
    df['sentiment'] = sentiment_df['smoothed_index'].reindex(df.index)
    df['MA50'] = df['Close'].rolling(50).mean()
    df = df.dropna()
    df = df[start_date:end_date]
    return df


def run_threshold_staged_train(df, and_threshold, or_threshold):
    cash = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    bought_levels = set()
    initial_capital_for_batch = cash

    for i in range(len(df)):
        current_price = df['Close'].iloc[i]
        current_sentiment = df['sentiment'].iloc[i]
        current_ma50 = df['MA50'].iloc[i]

        if position > 0:
            sell = False
            if current_sentiment > or_threshold:
                sell = True
            elif current_sentiment > and_threshold and current_price < current_ma50:
                sell = True
            if sell:
                cash += position * current_price * (1 - SLIPPAGE) * (1 - COMMISSION)
                position = 0
                entry_price = 0
                bought_levels = set()
                initial_capital_for_batch = cash

        for level_idx, threshold in enumerate(BUY_THRESHOLDS):
            if level_idx not in bought_levels and current_sentiment < threshold:
                buy_price = current_price * (1 + SLIPPAGE) * (1 + COMMISSION)
                target_value = initial_capital_for_batch * BATCH_PCT
                shares = int(target_value / buy_price)
                if shares > 0 and cash >= shares * buy_price:
                    buy_cost = shares * buy_price
                    if position > 0:
                        total_cost = entry_price * position + buy_cost
                        position += shares
                        entry_price = total_cost / position
                    else:
                        position = shares
                        entry_price = buy_price
                    cash -= buy_cost
                    bought_levels.add(level_idx)

    return cash + position * df['Close'].iloc[-1] if len(df) > 0 else cash


def grid_search(train_df):
    best_return = -float('inf')
    best_params = None
    all_results = []

    for and_t in AND_SELL_RANGE:
        for or_t in OR_SELL_RANGE:
            fv = run_threshold_staged_train(train_df, and_t, or_t)
            ret = (fv / INITIAL_CAPITAL - 1) * 100
            all_results.append((and_t, or_t, ret))
            if ret > best_return:
                best_return = ret
                best_params = (and_t, or_t)

    return best_params, best_return, all_results


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("=" * 70)
    print("卖出参数网格搜索 - 全部窗口 + W2026")
    print("=" * 70)
    print(f"AND搜索: {AND_SELL_RANGE}")
    print(f"OR搜索:  {OR_SELL_RANGE}")
    print(f"训练期买入: 阈值分批 {BUY_THRESHOLDS}, 每档{BATCH_PCT*100:.0f}%")

    rows = []

    for symbol in SYMBOLS:
        print(f"\n{'='*70}")
        print(f"  {symbol}")
        print(f"{'='*70}")

        price_df = load_price(symbol)
        sentiment_df = load_sentiment_s3(symbol)

        for window in WINDOWS:
            wname = window['name']
            train_start, train_end = window['train']
            train_df = prepare_data(price_df, sentiment_df, train_start, train_end)

            if len(train_df) < 100:
                print(f"  {wname}: 数据不足")
                continue

            best_params, best_return, all_results = grid_search(train_df)
            and_t, or_t = best_params

            # 次优参数
            sorted_results = sorted(all_results, key=lambda x: x[2], reverse=True)
            second = sorted_results[1] if len(sorted_results) > 1 else (0, 0, 0)

            print(f"  {wname} (train {train_start[:4]}-{train_end[:4]}): "
                  f"AND>{and_t}, OR>{or_t} → +{best_return:.1f}%  "
                  f"(次优: AND>{second[0]}, OR>{second[1]} → +{second[2]:.1f}%)")

            rows.append({
                'symbol': symbol,
                'window': wname,
                'train_period': f"{train_start[:4]}-{train_end[:4]}",
                'and_threshold': and_t,
                'or_threshold': or_t,
                'train_return': round(best_return, 1)
            })

    # 汇总表
    print("\n" + "=" * 70)
    print("2026年推荐卖出参数 (W2026)")
    print("=" * 70)

    header = f"{'股票':<8} {'AND阈值':>8} {'OR阈值':>8} {'训练收益':>10}"
    print(header)
    print("-" * 40)

    for row in rows:
        if row['window'] == 'W2026':
            print(f"{row['symbol']:<8} {'>'+str(row['and_threshold']):>8} "
                  f"{'>'+str(row['or_threshold']):>8} {row['train_return']:>+9.1f}%")

    # 全部窗口参数表
    print("\n" + "=" * 70)
    print("全部窗口卖出参数")
    print("=" * 70)

    header = f"{'股票':<8}"
    for w in WINDOWS:
        header += f" {w['name']:>12}"
    print(header)
    print("-" * (8 + 13 * len(WINDOWS)))

    for symbol in SYMBOLS:
        row = f"{symbol:<8}"
        for w in WINDOWS:
            match = [r for r in rows if r['symbol'] == symbol and r['window'] == w['name']]
            if match:
                row += f" A{match[0]['and_threshold']:>2}/O{match[0]['or_threshold']:<2}  "
            else:
                row += f" {'N/A':>10}  "
        print(row)

    # 保存CSV
    df = pd.DataFrame(rows)
    fname = f'sell_params_all_windows_{timestamp}.csv'
    df.to_csv(fname, index=False)
    print(f"\n输出: {fname}")


if __name__ == "__main__":
    main()
