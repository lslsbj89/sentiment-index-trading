"""
参数季度更新脚本
=========================================
用法:
  python3 retrain_params.py              # 默认训练期: 最近4年
  python3 retrain_params.py 2022 2025    # 指定训练期: 2022-2025

原理:
  在训练期用间隔分批买入 (sentiment < 0, 每7天买20%, 回升全买)
  网格搜索 AND × OR = 25 组合, 选训练收益最高的卖出参数
  训练买入策略与生产测试策略一致 (interval buy training)

输出:
  params_YYYYMMDD.csv  (买入+卖出参数)
"""

import sys
sys.path.insert(0, '/Users/sc2025/Desktop/Claude/Quantrade/src')

import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime
from data_loader import DataLoader

# ============================================================
# 配置 (买入参数固定, 仅搜索卖出参数)
# ============================================================

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

# 训练期买入策略 (间隔分批, 与生产测试策略一致)
INTERVAL_BUY_THRESHOLD = 0
INTERVAL_DAYS = 7
INTERVAL_BATCH_PCT = 0.20

# 卖出参数搜索范围
AND_SELL_RANGE = [5, 10, 15, 20, 25]
OR_SELL_RANGE = [30, 40, 50, 55, 60]

SYMBOLS = ["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META"]

# 个股最优买入阈值 (Walk-Forward回测结果)
OPTIMAL_BUY_THRESHOLDS = {
    "NVDA": 0,
    "TSLA": -15,
    "AAPL": 5,
    "MSFT": 0,
    "GOOGL": -5,
    "AMZN": 0,
    "META": 0,
}


# ============================================================
# 数据加载
# ============================================================

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


# ============================================================
# 训练期回测 (间隔分批, 与生产测试策略一致)
# ============================================================

def run_train(df, and_threshold, or_threshold):
    cash = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    in_buy_mode = False
    last_buy_date = None
    buy_base = None
    batches = 0

    for i in range(len(df)):
        dt = df.index[i]
        price = df['Close'].iloc[i]
        sent = df['sentiment'].iloc[i]
        ma50 = df['MA50'].iloc[i]

        # Sell
        if position > 0:
            if sent > or_threshold or (sent > and_threshold and price < ma50):
                cash += position * price * (1 - SLIPPAGE) * (1 - COMMISSION)
                position = 0
                entry_price = 0
                in_buy_mode = False
                last_buy_date = None
                buy_base = None
                batches = 0

        # Buy (interval logic)
        if position == 0 or in_buy_mode:
            if not in_buy_mode and sent < INTERVAL_BUY_THRESHOLD:
                in_buy_mode = True
                buy_base = cash + position * price
                batches = 0
                last_buy_date = None

            if in_buy_mode:
                should_buy = False
                buy_all = False
                if last_buy_date is None:
                    should_buy = True
                elif sent >= INTERVAL_BUY_THRESHOLD:
                    should_buy = True
                    buy_all = True
                elif (dt - last_buy_date).days >= INTERVAL_DAYS:
                    batches += 1
                    should_buy = True

                if should_buy:
                    bp = price * (1 + SLIPPAGE) * (1 + COMMISSION)
                    shares = int(cash / bp) if buy_all else int(buy_base * INTERVAL_BATCH_PCT / bp)
                    if shares > 0 and cash >= shares * bp:
                        cost = shares * bp
                        entry_price = (entry_price * position + cost) / (position + shares) if position > 0 else bp
                        position += shares
                        cash -= cost
                        last_buy_date = dt
                    if buy_all:
                        in_buy_mode = False
                        buy_base = None
                        batches = 0

    return cash + position * df['Close'].iloc[-1] if len(df) > 0 else cash


def grid_search(train_df):
    best_return = -float('inf')
    best_params = None
    results = []

    for and_t in AND_SELL_RANGE:
        for or_t in OR_SELL_RANGE:
            fv = run_train(train_df, and_t, or_t)
            ret = (fv / INITIAL_CAPITAL - 1) * 100
            results.append((and_t, or_t, ret))
            if ret > best_return:
                best_return = ret
                best_params = (and_t, or_t)

    return best_params, best_return, results


# ============================================================
# 主函数
# ============================================================

def main():
    # 解析命令行参数
    if len(sys.argv) >= 3:
        train_start_year = int(sys.argv[1])
        train_end_year = int(sys.argv[2])
    else:
        # 默认: 最近4年
        now = datetime.now()
        train_end_year = now.year - 1
        train_start_year = train_end_year - 3

    train_start = f"{train_start_year}-01-01"
    train_end = f"{train_end_year}-12-31"
    timestamp = datetime.now().strftime('%Y%m%d')

    print("=" * 60)
    print(f"卖出参数训练 | 训练期: {train_start} ~ {train_end}")
    print("=" * 60)
    print(f"AND搜索: {AND_SELL_RANGE}")
    print(f"OR搜索:  {OR_SELL_RANGE}")

    rows = []
    for symbol in SYMBOLS:
        price_df = load_price(symbol)
        sentiment_df = load_sentiment_s3(symbol)
        train_df = prepare_data(price_df, sentiment_df, train_start, train_end)

        if len(train_df) < 100:
            print(f"  {symbol}: 数据不足, 跳过")
            continue

        best_params, best_return, results = grid_search(train_df)
        and_t, or_t = best_params

        # 次优
        top2 = sorted(results, key=lambda x: x[2], reverse=True)[:2]

        print(f"  {symbol}: AND>{and_t}, OR>{or_t} → +{best_return:.1f}%"
              f"  (次优: A>{top2[1][0]}/O>{top2[1][1]} +{top2[1][2]:.1f}%)")

        rows.append({
            'symbol': symbol,
            'buy_threshold': INTERVAL_BUY_THRESHOLD,
            'optimal_buy': OPTIMAL_BUY_THRESHOLDS[symbol],
            'and_threshold': and_t,
            'or_threshold': or_t,
            'train_return': round(best_return, 1),
            'train_period': f"{train_start_year}-{train_end_year}"
        })

    # 输出
    print("\n" + "=" * 70)
    print("参数汇总")
    print("=" * 70)
    print(f"{'股票':<8} {'Buy<':>6} {'Opt<':>6} {'AND':>6} {'OR':>6} {'训练收益':>10}")
    print("-" * 50)
    for r in rows:
        print(f"{r['symbol']:<8} {r['buy_threshold']:>6} {r['optimal_buy']:>6} "
              f"{'>'+str(r['and_threshold']):>6} "
              f"{'>'+str(r['or_threshold']):>6} {r['train_return']:>+9.1f}%")

    fname = f"params_{timestamp}.csv"
    pd.DataFrame(rows).to_csv(fname, index=False)
    print(f"\n输出: {fname}")
    print(f"下次更新: 3个月后重新运行本脚本")


if __name__ == "__main__":
    main()
