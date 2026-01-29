"""
卖出策略对比回测
=====================================
对比多种卖出策略的效果:
A. 原策略: OR/AND条件
B. 原策略 + 硬性止损
C. 原策略 + 移动止盈
D. 原策略 + 分批卖出
E. 综合策略 (止损+止盈+情绪)

Walk-Forward验证: 4年训练 + 1年测试
"""

import sys
sys.path.insert(0, '/Users/sc2025/Desktop/Claude/Quantrade/src')

import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime
from data_loader import DataLoader

# ============================================================
# 配置
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

# 阈值分批买入参数 (统一)
BUY_THRESHOLDS = [5, 0, -5, -10]
BATCH_PCT = 0.25

# 卖出参数搜索范围
AND_SELL_RANGE = [5, 10, 15, 20, 25]
OR_SELL_RANGE = [30, 40, 50, 55, 60]

# 新增卖出参数
STOP_LOSS_PCT = 0.15  # 硬性止损: 亏损15%
TRAILING_PROFIT_TRIGGER = 0.30  # 移动止盈触发: 盈利30%
TRAILING_STOP_PCT = 0.10  # 移动止盈回撤: 10%

# 分批卖出阈值
SELL_THRESHOLDS = [15, 25, 40, 55]
SELL_BATCH_PCT = 0.25

WINDOWS = [
    {"name": "W2020", "train": ("2016-01-01", "2019-12-31"), "test": ("2020-01-01", "2020-12-31")},
    {"name": "W2021", "train": ("2017-01-01", "2020-12-31"), "test": ("2021-01-01", "2021-12-31")},
    {"name": "W2022", "train": ("2018-01-01", "2021-12-31"), "test": ("2022-01-01", "2022-12-31")},
    {"name": "W2023", "train": ("2019-01-01", "2022-12-31"), "test": ("2023-01-01", "2023-12-31")},
    {"name": "W2024", "train": ("2020-01-01", "2023-12-31"), "test": ("2024-01-01", "2024-12-31")},
    {"name": "W2025", "train": ("2021-01-01", "2024-12-31"), "test": ("2025-01-01", "2025-12-31")},
]


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
# 策略A: 原策略 (OR/AND条件)
# ============================================================

def strategy_original(df, and_threshold, or_threshold, cash, position, entry_price, bought_levels):
    """原策略: 只有OR/AND卖出条件"""
    if bought_levels is None:
        bought_levels = set()

    trades = []
    initial_capital_for_batch = cash + position * df['Close'].iloc[0] if len(df) > 0 and position > 0 else cash

    for i in range(len(df)):
        current_date = df.index[i]
        current_price = df['Close'].iloc[i]
        current_sentiment = df['sentiment'].iloc[i]
        current_ma50 = df['MA50'].iloc[i]

        # 卖出逻辑: 只有OR/AND
        if position > 0:
            sell_signal = False
            sell_reason = ""

            if current_sentiment > or_threshold:
                sell_signal = True
                sell_reason = f"OR: sent {current_sentiment:.1f}>{or_threshold}"
            elif current_sentiment > and_threshold and current_price < current_ma50:
                sell_signal = True
                sell_reason = f"AND: sent {current_sentiment:.1f}>{and_threshold}"

            if sell_signal:
                sell_price = current_price * (1 - SLIPPAGE) * (1 - COMMISSION)
                profit_pct = (sell_price - entry_price) / entry_price * 100
                cash += position * sell_price
                trades.append({'type': 'SELL', 'date': current_date, 'price': current_price,
                              'reason': sell_reason, 'profit_pct': profit_pct})
                position = 0
                entry_price = 0
                bought_levels = set()
                initial_capital_for_batch = cash

        # 买入逻辑
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
                    trades.append({'type': 'BUY', 'date': current_date, 'price': current_price,
                                  'reason': f"Batch{level_idx+1}", 'batch': level_idx+1})

    final_value = cash + position * df['Close'].iloc[-1] if len(df) > 0 else cash
    return final_value, trades, cash, position, entry_price, bought_levels


# ============================================================
# 策略B: 原策略 + 硬性止损
# ============================================================

def strategy_with_stop_loss(df, and_threshold, or_threshold, cash, position, entry_price, bought_levels,
                            stop_loss_pct=STOP_LOSS_PCT):
    """原策略 + 硬性止损"""
    if bought_levels is None:
        bought_levels = set()

    trades = []
    initial_capital_for_batch = cash + position * df['Close'].iloc[0] if len(df) > 0 and position > 0 else cash

    for i in range(len(df)):
        current_date = df.index[i]
        current_price = df['Close'].iloc[i]
        current_sentiment = df['sentiment'].iloc[i]
        current_ma50 = df['MA50'].iloc[i]

        # 卖出逻辑
        if position > 0:
            sell_signal = False
            sell_reason = ""
            current_profit_pct = (current_price - entry_price) / entry_price

            # 1. 硬性止损 (优先)
            if current_profit_pct < -stop_loss_pct:
                sell_signal = True
                sell_reason = f"STOP_LOSS: {current_profit_pct*100:.1f}%<-{stop_loss_pct*100:.0f}%"
            # 2. OR条件
            elif current_sentiment > or_threshold:
                sell_signal = True
                sell_reason = f"OR: sent {current_sentiment:.1f}>{or_threshold}"
            # 3. AND条件
            elif current_sentiment > and_threshold and current_price < current_ma50:
                sell_signal = True
                sell_reason = f"AND: sent {current_sentiment:.1f}>{and_threshold}"

            if sell_signal:
                sell_price = current_price * (1 - SLIPPAGE) * (1 - COMMISSION)
                profit_pct = (sell_price - entry_price) / entry_price * 100
                cash += position * sell_price
                trades.append({'type': 'SELL', 'date': current_date, 'price': current_price,
                              'reason': sell_reason, 'profit_pct': profit_pct})
                position = 0
                entry_price = 0
                bought_levels = set()
                initial_capital_for_batch = cash

        # 买入逻辑 (同原策略)
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
                    trades.append({'type': 'BUY', 'date': current_date, 'price': current_price,
                                  'reason': f"Batch{level_idx+1}", 'batch': level_idx+1})

    final_value = cash + position * df['Close'].iloc[-1] if len(df) > 0 else cash
    return final_value, trades, cash, position, entry_price, bought_levels


# ============================================================
# 策略C: 原策略 + 移动止盈
# ============================================================

def strategy_with_trailing_stop(df, and_threshold, or_threshold, cash, position, entry_price, bought_levels,
                                 trailing_trigger=TRAILING_PROFIT_TRIGGER, trailing_stop=TRAILING_STOP_PCT):
    """原策略 + 移动止盈"""
    if bought_levels is None:
        bought_levels = set()

    trades = []
    initial_capital_for_batch = cash + position * df['Close'].iloc[0] if len(df) > 0 and position > 0 else cash
    max_profit_pct = 0  # 追踪最高盈利

    for i in range(len(df)):
        current_date = df.index[i]
        current_price = df['Close'].iloc[i]
        current_sentiment = df['sentiment'].iloc[i]
        current_ma50 = df['MA50'].iloc[i]

        # 卖出逻辑
        if position > 0:
            sell_signal = False
            sell_reason = ""
            current_profit_pct = (current_price - entry_price) / entry_price

            # 更新最高盈利
            if current_profit_pct > max_profit_pct:
                max_profit_pct = current_profit_pct

            # 1. 移动止盈
            if max_profit_pct > trailing_trigger:
                trailing_line = max_profit_pct - trailing_stop
                if current_profit_pct < trailing_line:
                    sell_signal = True
                    sell_reason = f"TRAILING: {current_profit_pct*100:.1f}%<{trailing_line*100:.1f}% (peak {max_profit_pct*100:.1f}%)"

            # 2. OR条件
            if not sell_signal and current_sentiment > or_threshold:
                sell_signal = True
                sell_reason = f"OR: sent {current_sentiment:.1f}>{or_threshold}"

            # 3. AND条件
            if not sell_signal and current_sentiment > and_threshold and current_price < current_ma50:
                sell_signal = True
                sell_reason = f"AND: sent {current_sentiment:.1f}>{and_threshold}"

            if sell_signal:
                sell_price = current_price * (1 - SLIPPAGE) * (1 - COMMISSION)
                profit_pct = (sell_price - entry_price) / entry_price * 100
                cash += position * sell_price
                trades.append({'type': 'SELL', 'date': current_date, 'price': current_price,
                              'reason': sell_reason, 'profit_pct': profit_pct})
                position = 0
                entry_price = 0
                bought_levels = set()
                initial_capital_for_batch = cash
                max_profit_pct = 0

        # 买入逻辑 (同原策略)
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
                    trades.append({'type': 'BUY', 'date': current_date, 'price': current_price,
                                  'reason': f"Batch{level_idx+1}", 'batch': level_idx+1})
                    max_profit_pct = 0  # 重置

    final_value = cash + position * df['Close'].iloc[-1] if len(df) > 0 else cash
    return final_value, trades, cash, position, entry_price, bought_levels


# ============================================================
# 策略D: 综合策略 (止损 + 止盈 + 情绪)
# ============================================================

def strategy_combined(df, and_threshold, or_threshold, cash, position, entry_price, bought_levels,
                      stop_loss_pct=STOP_LOSS_PCT, trailing_trigger=TRAILING_PROFIT_TRIGGER,
                      trailing_stop=TRAILING_STOP_PCT):
    """综合策略: 止损 + 移动止盈 + OR/AND"""
    if bought_levels is None:
        bought_levels = set()

    trades = []
    initial_capital_for_batch = cash + position * df['Close'].iloc[0] if len(df) > 0 and position > 0 else cash
    max_profit_pct = 0

    for i in range(len(df)):
        current_date = df.index[i]
        current_price = df['Close'].iloc[i]
        current_sentiment = df['sentiment'].iloc[i]
        current_ma50 = df['MA50'].iloc[i]

        # 卖出逻辑
        if position > 0:
            sell_signal = False
            sell_reason = ""
            current_profit_pct = (current_price - entry_price) / entry_price

            # 更新最高盈利
            if current_profit_pct > max_profit_pct:
                max_profit_pct = current_profit_pct

            # 1. 硬性止损 (最高优先级)
            if current_profit_pct < -stop_loss_pct:
                sell_signal = True
                sell_reason = f"STOP_LOSS: {current_profit_pct*100:.1f}%"

            # 2. 移动止盈
            if not sell_signal and max_profit_pct > trailing_trigger:
                trailing_line = max_profit_pct - trailing_stop
                if current_profit_pct < trailing_line:
                    sell_signal = True
                    sell_reason = f"TRAILING: {current_profit_pct*100:.1f}%"

            # 3. OR条件
            if not sell_signal and current_sentiment > or_threshold:
                sell_signal = True
                sell_reason = f"OR: sent {current_sentiment:.1f}>{or_threshold}"

            # 4. AND条件
            if not sell_signal and current_sentiment > and_threshold and current_price < current_ma50:
                sell_signal = True
                sell_reason = f"AND: sent {current_sentiment:.1f}>{and_threshold}"

            if sell_signal:
                sell_price = current_price * (1 - SLIPPAGE) * (1 - COMMISSION)
                profit_pct = (sell_price - entry_price) / entry_price * 100
                cash += position * sell_price
                trades.append({'type': 'SELL', 'date': current_date, 'price': current_price,
                              'reason': sell_reason, 'profit_pct': profit_pct})
                position = 0
                entry_price = 0
                bought_levels = set()
                initial_capital_for_batch = cash
                max_profit_pct = 0

        # 买入逻辑 (同原策略)
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
                    trades.append({'type': 'BUY', 'date': current_date, 'price': current_price,
                                  'reason': f"Batch{level_idx+1}", 'batch': level_idx+1})
                    max_profit_pct = 0

    final_value = cash + position * df['Close'].iloc[-1] if len(df) > 0 else cash
    return final_value, trades, cash, position, entry_price, bought_levels


# ============================================================
# 网格搜索 (只用原策略找参数)
# ============================================================

def grid_search(train_df):
    best_return = -float('inf')
    best_params = None
    for and_t in AND_SELL_RANGE:
        for or_t in OR_SELL_RANGE:
            final_value, _, _, _, _, _ = strategy_original(
                train_df, and_t, or_t, INITIAL_CAPITAL, 0, 0, None
            )
            ret = (final_value / INITIAL_CAPITAL - 1) * 100
            if ret > best_return:
                best_return = ret
                best_params = (and_t, or_t)
    return best_params, best_return


# ============================================================
# Walk-Forward 对比
# ============================================================

def run_comparison(symbol):
    """对单个股票运行所有策略对比"""
    print(f"\n{'='*80}")
    print(f"  {symbol} - 卖出策略对比")
    print(f"{'='*80}")

    price_df = load_price(symbol)
    sentiment_df = load_sentiment_s3(symbol)

    strategies = {
        'A.Original': strategy_original,
        'B.+StopLoss': strategy_with_stop_loss,
        'C.+Trailing': strategy_with_trailing_stop,
        'D.Combined': strategy_combined,
    }

    # 每个策略独立运行
    results = {name: {'windows': [], 'final_value': 0, 'trades': []} for name in strategies}

    for name in strategies:
        cash = INITIAL_CAPITAL
        position = 0
        entry_price = 0
        bought_levels = None
        all_trades = []

        for window in WINDOWS:
            window_name = window['name']
            train_start, train_end = window['train']
            test_start, test_end = window['test']

            train_df = prepare_data(price_df, sentiment_df, train_start, train_end)
            test_df = prepare_data(price_df, sentiment_df, test_start, test_end)

            if len(train_df) < 100 or len(test_df) < 10:
                continue

            # 训练期找参数 (使用原策略)
            best_params, _ = grid_search(train_df)
            and_t, or_t = best_params

            # 测试期开始资产
            start_value = cash + position * test_df['Close'].iloc[0] if position > 0 else cash

            # 运行策略
            strategy_func = strategies[name]
            if name == 'A.Original':
                final_value, trades, cash, position, entry_price, bought_levels = strategy_func(
                    test_df, and_t, or_t, cash, position, entry_price, bought_levels
                )
            else:
                final_value, trades, cash, position, entry_price, bought_levels = strategy_func(
                    test_df, and_t, or_t, cash, position, entry_price, bought_levels
                )

            test_return = (final_value / start_value - 1) * 100 if start_value > 0 else 0

            results[name]['windows'].append({
                'window': window_name,
                'start_value': start_value,
                'end_value': final_value,
                'return': test_return,
                'trades': len(trades)
            })

            for t in trades:
                t['window'] = window_name
                all_trades.append(t)

        results[name]['final_value'] = final_value
        results[name]['trades'] = all_trades

    # 打印对比结果
    print(f"\n  各窗口收益对比:")
    print(f"  {'窗口':<8}", end='')
    for name in strategies:
        print(f"{name:>15}", end='')
    print()
    print(f"  {'-'*70}")

    for i, window in enumerate(WINDOWS):
        print(f"  {window['name']:<8}", end='')
        for name in strategies:
            if i < len(results[name]['windows']):
                ret = results[name]['windows'][i]['return']
                print(f"{ret:>+14.1f}%", end='')
            else:
                print(f"{'N/A':>15}", end='')
        print()

    print(f"  {'-'*70}")
    print(f"  {'总收益':<8}", end='')
    for name in strategies:
        total_ret = (results[name]['final_value'] / INITIAL_CAPITAL - 1) * 100
        print(f"{total_ret:>+14.1f}%", end='')
    print()

    # 统计卖出原因
    print(f"\n  卖出原因统计:")
    for name in strategies:
        sells = [t for t in results[name]['trades'] if t['type'] == 'SELL']
        print(f"\n  {name}:")
        reasons = {}
        for t in sells:
            reason_type = t['reason'].split(':')[0]
            reasons[reason_type] = reasons.get(reason_type, 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}次")

    # 2022年特别分析
    print(f"\n  2022年熊市表现:")
    for name in strategies:
        for w in results[name]['windows']:
            if w['window'] == 'W2022':
                print(f"    {name}: {w['return']:+.1f}% ({w['trades']}笔交易)")

    return results


# ============================================================
# 主函数
# ============================================================

def main():
    print("="*80)
    print("  卖出策略对比回测")
    print("="*80)
    print(f"\n  策略说明:")
    print(f"  A.Original: 原策略 (只有OR/AND条件)")
    print(f"  B.+StopLoss: 原策略 + 硬性止损 ({STOP_LOSS_PCT*100:.0f}%)")
    print(f"  C.+Trailing: 原策略 + 移动止盈 (盈利>{TRAILING_PROFIT_TRIGGER*100:.0f}%后回撤{TRAILING_STOP_PCT*100:.0f}%)")
    print(f"  D.Combined: 综合策略 (止损+止盈+情绪)")

    symbols = ["NVDA", "TSLA", "META", "AAPL", "MSFT", "GOOGL", "AMZN"]
    all_results = {}

    for symbol in symbols:
        all_results[symbol] = run_comparison(symbol)

    # 总结
    print(f"\n\n{'='*80}")
    print(f"  七股汇总对比")
    print(f"{'='*80}")

    print(f"\n  {'股票':<8} {'A.Original':>14} {'B.+StopLoss':>14} {'C.+Trailing':>14} {'D.Combined':>14} {'最优策略':>12}")
    print(f"  {'-'*80}")

    for symbol in symbols:
        results = all_results[symbol]
        returns = {}
        for name in results:
            returns[name] = (results[name]['final_value'] / INITIAL_CAPITAL - 1) * 100

        best = max(returns, key=returns.get)
        print(f"  {symbol:<8}", end='')
        for name in ['A.Original', 'B.+StopLoss', 'C.+Trailing', 'D.Combined']:
            ret = returns[name]
            marker = '*' if name == best else ' '
            print(f"{ret:>+13.1f}%{marker}", end='')
        print(f"  {best:>12}")

    # 2022年熊市对比
    print(f"\n  2022年熊市表现:")
    print(f"  {'股票':<8} {'A.Original':>14} {'B.+StopLoss':>14} {'C.+Trailing':>14} {'D.Combined':>14}")
    print(f"  {'-'*70}")

    for symbol in symbols:
        results = all_results[symbol]
        print(f"  {symbol:<8}", end='')
        for name in ['A.Original', 'B.+StopLoss', 'C.+Trailing', 'D.Combined']:
            for w in results[name]['windows']:
                if w['window'] == 'W2022':
                    print(f"{w['return']:>+14.1f}%", end='')
                    break
        print()


if __name__ == "__main__":
    main()
