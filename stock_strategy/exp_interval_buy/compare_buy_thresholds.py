"""
间隔分批买入 - 买入阈值对比实验
=====================================
固定: interval=7天, batch=20%, 卖出参数由原版网格搜索确定
变量: buy_threshold = [-15, -10, -5, 0, 5]

对比不同买入阈值对7只股票收益的影响
"""

import sys
sys.path.insert(0, '/Users/sc2025/Desktop/Claude/Quantrade/src')

import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime
from data_loader import DataLoader

# ============================================================
# 配置参数
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

# 固定参数
INTERVAL_DAYS = 7
BATCH_PCT = 0.20

# 对比的买入阈值
BUY_THRESHOLD_LIST = [-15, -10, -5, 0, 5]

# 训练期原版阈值分批参数 (用于网格搜索卖出参数)
BUY_THRESHOLDS_TRAIN = [5, 0, -5, -10]
BATCH_PCT_TRAIN = 0.25
AND_SELL_RANGE = [5, 10, 15, 20, 25]
OR_SELL_RANGE = [30, 40, 50, 55, 60]

WINDOWS = [
    {"name": "W2020", "train": ("2016-01-01", "2019-12-31"), "test": ("2020-01-01", "2020-12-31")},
    {"name": "W2021", "train": ("2017-01-01", "2020-12-31"), "test": ("2021-01-01", "2021-12-31")},
    {"name": "W2022", "train": ("2018-01-01", "2021-12-31"), "test": ("2022-01-01", "2022-12-31")},
    {"name": "W2023", "train": ("2019-01-01", "2022-12-31"), "test": ("2023-01-01", "2023-12-31")},
    {"name": "W2024", "train": ("2020-01-01", "2023-12-31"), "test": ("2024-01-01", "2024-12-31")},
    {"name": "W2025", "train": ("2021-01-01", "2024-12-31"), "test": ("2025-01-01", "2025-12-31")},
]

SYMBOLS = ["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META"]


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
# 原版阈值分批 (训练期网格搜索)
# ============================================================

def run_threshold_staged_train(df, and_threshold, or_threshold):
    """训练期原版策略, 用于网格搜索卖出参数"""
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

        for level_idx, threshold in enumerate(BUY_THRESHOLDS_TRAIN):
            if level_idx not in bought_levels and current_sentiment < threshold:
                buy_price = current_price * (1 + SLIPPAGE) * (1 + COMMISSION)
                target_value = initial_capital_for_batch * BATCH_PCT_TRAIN
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
    for and_t in AND_SELL_RANGE:
        for or_t in OR_SELL_RANGE:
            fv = run_threshold_staged_train(train_df, and_t, or_t)
            ret = (fv / INITIAL_CAPITAL - 1) * 100
            if ret > best_return:
                best_return = ret
                best_params = (and_t, or_t)
    return best_params, best_return


# ============================================================
# 间隔分批买入 (测试期)
# ============================================================

def run_interval_buy(df, buy_threshold, and_threshold, or_threshold,
                     cash, position, entry_price,
                     in_buy_mode, last_buy_date, buy_base_capital, batches_bought):
    """间隔分批买入"""
    trades = []

    for i in range(len(df)):
        current_date = df.index[i]
        current_price = df['Close'].iloc[i]
        current_sentiment = df['sentiment'].iloc[i]
        current_ma50 = df['MA50'].iloc[i]

        # 卖出
        if position > 0:
            sell_signal = False
            sell_reason = ""
            if current_sentiment > or_threshold:
                sell_signal = True
                sell_reason = f"OR: sentiment {current_sentiment:.1f} > {or_threshold}"
            elif current_sentiment > and_threshold and current_price < current_ma50:
                sell_signal = True
                sell_reason = f"AND: sentiment {current_sentiment:.1f} > {and_threshold} & price < MA50"

            if sell_signal:
                sell_price = current_price * (1 - SLIPPAGE) * (1 - COMMISSION)
                profit_pct = (sell_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
                cash += position * sell_price
                trades.append({
                    'type': 'SELL', 'date': current_date, 'price': current_price,
                    'shares': position, 'sentiment': current_sentiment,
                    'reason': sell_reason, 'profit_pct': profit_pct
                })
                position = 0
                entry_price = 0
                in_buy_mode = False
                last_buy_date = None
                buy_base_capital = None
                batches_bought = 0

        # 买入
        if position == 0 or in_buy_mode:
            if not in_buy_mode and current_sentiment < buy_threshold:
                in_buy_mode = True
                buy_base_capital = cash + position * current_price
                batches_bought = 0
                last_buy_date = None

            if in_buy_mode:
                should_buy = False
                buy_all = False

                if last_buy_date is None:
                    should_buy = True
                elif current_sentiment >= buy_threshold:
                    should_buy = True
                    buy_all = True
                elif (current_date - last_buy_date).days >= INTERVAL_DAYS:
                    should_buy = True
                    batches_bought += 1

                if should_buy:
                    buy_price = current_price * (1 + SLIPPAGE) * (1 + COMMISSION)
                    if buy_all:
                        shares = int(cash / buy_price)
                    else:
                        shares = int(buy_base_capital * BATCH_PCT / buy_price)

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
                        last_buy_date = current_date
                        trades.append({
                            'type': 'BUY', 'date': current_date, 'price': current_price,
                            'shares': shares, 'sentiment': current_sentiment,
                        })

                    if buy_all:
                        in_buy_mode = False
                        buy_base_capital = None
                        batches_bought = 0

    final_value = cash + position * df['Close'].iloc[-1] if len(df) > 0 else cash
    return final_value, trades, cash, position, entry_price, in_buy_mode, last_buy_date, buy_base_capital, batches_bought


# ============================================================
# Walk-Forward (单个阈值)
# ============================================================

def run_walk_forward(symbol, buy_threshold, price_df, sentiment_df):
    """对单个股票+单个买入阈值运行Walk-Forward"""
    cash = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    in_buy_mode = False
    last_buy_date = None
    buy_base_capital = None
    batches_bought = 0

    all_trades = []
    window_results = []

    for window in WINDOWS:
        train_start, train_end = window['train']
        test_start, test_end = window['test']

        train_df = prepare_data(price_df, sentiment_df, train_start, train_end)
        test_df = prepare_data(price_df, sentiment_df, test_start, test_end)

        if len(train_df) < 100 or len(test_df) < 10:
            continue

        best_params, _ = grid_search(train_df)
        and_t, or_t = best_params

        test_start_value = cash + position * test_df['Close'].iloc[0] if len(test_df) > 0 else cash

        (final_value, trades, cash, position, entry_price,
         in_buy_mode, last_buy_date, buy_base_capital, batches_bought) = run_interval_buy(
            test_df, buy_threshold, and_t, or_t,
            cash, position, entry_price,
            in_buy_mode, last_buy_date, buy_base_capital, batches_bought
        )

        test_return = (final_value / test_start_value - 1) * 100 if test_start_value > 0 else 0
        buy_count = len([t for t in trades if t['type'] == 'BUY'])
        sell_count = len([t for t in trades if t['type'] == 'SELL'])

        window_results.append({
            'window': window['name'],
            'and_t': and_t, 'or_t': or_t,
            'test_return': test_return,
            'buys': buy_count, 'sells': sell_count
        })

        for t in trades:
            t['window'] = window['name']
            t['symbol'] = symbol
        all_trades.extend(trades)

    total_return = (final_value / INITIAL_CAPITAL - 1) * 100
    all_sells = [t for t in all_trades if t['type'] == 'SELL']
    win_rate = len([t for t in all_sells if t['profit_pct'] > 0]) / len(all_sells) * 100 if all_sells else 0

    return {
        'total_return': total_return,
        'final_value': final_value,
        'win_rate': win_rate,
        'total_buys': sum(w['buys'] for w in window_results),
        'total_sells': sum(w['sells'] for w in window_results),
        'window_results': window_results
    }


# ============================================================
# 主函数
# ============================================================

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("="*70)
    print("间隔分批买入 - 买入阈值对比")
    print("="*70)
    print(f"买入阈值对比: {BUY_THRESHOLD_LIST}")
    print(f"固定参数: interval={INTERVAL_DAYS}天, batch={BATCH_PCT*100:.0f}%")
    print(f"卖出参数: 训练期网格搜索 (原版阈值分批)")
    print(f"初始资金: ${INITIAL_CAPITAL:,}")

    # 结果矩阵: results[symbol][buy_threshold]
    all_data = {}

    for symbol in SYMBOLS:
        print(f"\n{'='*70}")
        print(f"  {symbol}")
        print(f"{'='*70}")

        price_df = load_price(symbol)
        sentiment_df = load_sentiment_s3(symbol)

        symbol_results = {}
        for bt in BUY_THRESHOLD_LIST:
            result = run_walk_forward(symbol, bt, price_df, sentiment_df)
            symbol_results[bt] = result

            # 打印窗口详情
            print(f"\n  buy<{bt:>3}: ", end="")
            for w in result['window_results']:
                print(f"{w['window']}={w['test_return']:+.1f}% ", end="")
            print(f"| 总收益={result['total_return']:+.1f}% B{result['total_buys']}/S{result['total_sells']} W{result['win_rate']:.0f}%")

        all_data[symbol] = symbol_results

    # ========== 汇总表 ==========
    print("\n" + "="*70)
    print("总收益对比 (buy_threshold)")
    print("="*70)

    header = f"{'股票':<8}"
    for bt in BUY_THRESHOLD_LIST:
        header += f"{'buy<'+str(bt):>12}"
    header += f"{'最优':>12} {'最优阈值':>10}"
    print(header)
    print("-" * (8 + 12 * len(BUY_THRESHOLD_LIST) + 22))

    summary_rows = []
    for symbol in SYMBOLS:
        row = f"{symbol:<8}"
        returns = {}
        for bt in BUY_THRESHOLD_LIST:
            ret = all_data[symbol][bt]['total_return']
            returns[bt] = ret
            row += f"{ret:>+11.1f}%"
        best_bt = max(returns, key=returns.get)
        row += f"{returns[best_bt]:>+11.1f}% {'buy<'+str(best_bt):>10}"
        print(row)

        summary_rows.append({
            'symbol': symbol,
            **{f'buy<{bt}': all_data[symbol][bt]['total_return'] for bt in BUY_THRESHOLD_LIST},
            'best_threshold': best_bt,
            'best_return': returns[best_bt]
        })

    # 平均
    print("-" * (8 + 12 * len(BUY_THRESHOLD_LIST) + 22))
    avg_row = f"{'平均':<8}"
    avg_returns = {}
    for bt in BUY_THRESHOLD_LIST:
        avg = np.mean([all_data[s][bt]['total_return'] for s in SYMBOLS])
        avg_returns[bt] = avg
        avg_row += f"{avg:>+11.1f}%"
    best_avg_bt = max(avg_returns, key=avg_returns.get)
    avg_row += f"{avg_returns[best_avg_bt]:>+11.1f}% {'buy<'+str(best_avg_bt):>10}"
    print(avg_row)

    # ========== 买入/卖出次数对比 ==========
    print("\n" + "="*70)
    print("交易次数对比 (买入/卖出)")
    print("="*70)

    header = f"{'股票':<8}"
    for bt in BUY_THRESHOLD_LIST:
        header += f"{'buy<'+str(bt):>12}"
    print(header)
    print("-" * (8 + 12 * len(BUY_THRESHOLD_LIST)))

    for symbol in SYMBOLS:
        row = f"{symbol:<8}"
        for bt in BUY_THRESHOLD_LIST:
            r = all_data[symbol][bt]
            row += f"  B{r['total_buys']:>2}/S{r['total_sells']:<2}  "
        print(row)

    # ========== 保存CSV ==========
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f'compare_buy_thresholds_{timestamp}.csv', index=False)

    print(f"\n输出: compare_buy_thresholds_{timestamp}.csv")


if __name__ == "__main__":
    main()
