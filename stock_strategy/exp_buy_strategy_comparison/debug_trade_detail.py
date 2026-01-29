"""
详细交易过程 - 显示每笔交易和资产变化
"""

import sys
sys.path.insert(0, '/Users/sc2025/Desktop/Claude/Quantrade/src')

import pandas as pd
import numpy as np
import psycopg2
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

BUY_THRESHOLDS = [5, 0, -5, -10]
BATCH_PCT = 0.25

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


def run_threshold_staged_verbose(df, and_threshold, or_threshold,
                                  cash, position, entry_price, bought_levels, window_name):
    """带详细输出的阈值分批策略"""
    if bought_levels is None:
        bought_levels = set()

    trades = []
    initial_capital_for_batch = cash + position * df['Close'].iloc[0] if len(df) > 0 and position > 0 else cash

    print(f"\n  {'─'*90}")
    print(f"  {window_name} 测试期交易详情 (参数: AND>{and_threshold}, OR>{or_threshold})")
    print(f"  {'─'*90}")
    print(f"  分批基准资金: ${initial_capital_for_batch:,.2f} | 每批25% = ${initial_capital_for_batch * 0.25:,.2f}")
    print(f"  买入阈值: sent<5 (Batch1), sent<0 (Batch2), sent<-5 (Batch3), sent<-10 (Batch4)")
    print(f"  {'─'*90}")

    for i in range(len(df)):
        current_date = df.index[i]
        current_price = df['Close'].iloc[i]
        current_sentiment = df['sentiment'].iloc[i]
        current_ma50 = df['MA50'].iloc[i]

        traded_today = False

        # ========== 卖出逻辑 ==========
        if position > 0:
            sell_signal = False
            sell_reason = ""

            if current_sentiment > or_threshold:
                sell_signal = True
                sell_reason = f"OR条件: sent {current_sentiment:.1f} > {or_threshold}"
            elif current_sentiment > and_threshold and current_price < current_ma50:
                sell_signal = True
                sell_reason = f"AND条件: sent {current_sentiment:.1f} > {and_threshold} & price < MA50"

            if sell_signal:
                sell_price_actual = current_price * (1 - SLIPPAGE) * (1 - COMMISSION)
                profit_pct = (sell_price_actual - entry_price) / entry_price * 100 if entry_price > 0 else 0
                sell_value = position * sell_price_actual

                # 交易前状态
                before_cash = cash
                before_position = position
                before_total = before_cash + before_position * current_price

                cash += sell_value

                # 交易后状态
                after_cash = cash
                after_position = 0
                after_total = after_cash

                print(f"\n  📅 {current_date.strftime('%Y-%m-%d')} | 价格=${current_price:.2f} | 情绪={current_sentiment:.1f}")
                print(f"  🔴 SELL {position}股 × ${current_price:.2f} = ${sell_value:,.2f}")
                print(f"     {sell_reason}")
                print(f"     入场均价: ${entry_price:.2f} → 收益: {profit_pct:+.1f}%")
                print(f"     资产变化: ${before_total:,.2f} → ${after_total:,.2f}")

                trades.append({
                    'type': 'SELL',
                    'date': current_date,
                    'price': current_price,
                    'shares': position,
                    'sentiment': current_sentiment,
                    'reason': sell_reason,
                    'profit_pct': profit_pct,
                    'before_total': before_total,
                    'after_total': after_total
                })

                position = 0
                entry_price = 0
                bought_levels = set()
                initial_capital_for_batch = cash  # 重置分批基准
                traded_today = True

                print(f"     新分批基准: ${initial_capital_for_batch:,.2f} | 每批25% = ${initial_capital_for_batch * 0.25:,.2f}")

        # ========== 买入逻辑 ==========
        for level_idx, threshold in enumerate(BUY_THRESHOLDS):
            if level_idx not in bought_levels and current_sentiment < threshold:
                buy_price_actual = current_price * (1 + SLIPPAGE) * (1 + COMMISSION)
                target_value = initial_capital_for_batch * BATCH_PCT
                shares = int(target_value / buy_price_actual)

                if shares > 0 and cash >= shares * buy_price_actual:
                    buy_cost = shares * buy_price_actual

                    # 交易前状态
                    before_cash = cash
                    before_position = position
                    before_total = before_cash + before_position * current_price

                    # 更新持仓和均价
                    if position > 0:
                        total_cost = entry_price * position + buy_cost
                        position += shares
                        entry_price = total_cost / position
                    else:
                        position = shares
                        entry_price = buy_price_actual

                    cash -= buy_cost
                    bought_levels.add(level_idx)

                    # 交易后状态
                    after_cash = cash
                    after_position = position
                    after_total = after_cash + after_position * current_price

                    if not traded_today:
                        print(f"\n  📅 {current_date.strftime('%Y-%m-%d')} | 价格=${current_price:.2f} | 情绪={current_sentiment:.1f}")
                        traded_today = True

                    print(f"  🟢 BUY Batch{level_idx+1} (sent<{threshold}): {shares}股 × ${current_price:.2f} = ${buy_cost:,.2f}")
                    print(f"     已买档位: {sorted(bought_levels)} | 入场均价: ${entry_price:.2f}")
                    print(f"     资产: 现金${after_cash:,.2f} + 股票${after_position * current_price:,.2f} = ${after_total:,.2f}")

                    trades.append({
                        'type': 'BUY',
                        'date': current_date,
                        'price': current_price,
                        'shares': shares,
                        'sentiment': current_sentiment,
                        'reason': f"Batch{level_idx+1}: sent {current_sentiment:.1f} < {threshold}",
                        'batch': level_idx + 1,
                        'before_total': before_total,
                        'after_total': after_total,
                        'entry_price': entry_price
                    })

    # 期末状态
    end_price = df['Close'].iloc[-1]
    end_total = cash + position * end_price

    print(f"\n  {'─'*90}")
    print(f"  {window_name} 期末状态:")
    print(f"  现金: ${cash:,.2f}")
    print(f"  持仓: {position}股 × ${end_price:.2f} = ${position * end_price:,.2f}")
    print(f"  总资产: ${end_total:,.2f}")
    print(f"  {'─'*90}")

    return end_total, trades, cash, position, entry_price, bought_levels


def grid_search(train_df):
    best_return = -float('inf')
    best_params = None

    for and_t in AND_SELL_RANGE:
        for or_t in OR_SELL_RANGE:
            final_value, _, _, _, _, _ = run_threshold_staged_silent(
                train_df, and_t, or_t, INITIAL_CAPITAL, 0, 0, None
            )
            ret = (final_value / INITIAL_CAPITAL - 1) * 100

            if ret > best_return:
                best_return = ret
                best_params = (and_t, or_t)

    return best_params, best_return


def run_threshold_staged_silent(df, and_threshold, or_threshold,
                                 cash, position, entry_price, bought_levels):
    """静默版本用于网格搜索"""
    if bought_levels is None:
        bought_levels = set()

    initial_capital_for_batch = cash + position * df['Close'].iloc[0] if len(df) > 0 and position > 0 else cash

    for i in range(len(df)):
        current_price = df['Close'].iloc[i]
        current_sentiment = df['sentiment'].iloc[i]
        current_ma50 = df['MA50'].iloc[i]

        if position > 0:
            sell_signal = False
            if current_sentiment > or_threshold:
                sell_signal = True
            elif current_sentiment > and_threshold and current_price < current_ma50:
                sell_signal = True

            if sell_signal:
                sell_price = current_price * (1 - SLIPPAGE) * (1 - COMMISSION)
                cash += position * sell_price
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

    final_value = cash + position * df['Close'].iloc[-1] if len(df) > 0 else cash
    return final_value, [], cash, position, entry_price, bought_levels


def main():
    symbol = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"

    print("=" * 100)
    print(f"  {symbol} - 阈值分批策略详细交易过程")
    print("=" * 100)
    print(f"\n  策略说明:")
    print(f"  • 买入: 情绪 < 5/0/-5/-10 时分4批买入，每批25%")
    print(f"  • 卖出: OR条件(无条件) 或 AND条件(需价格<MA50)")
    print(f"  • 初始资金: ${INITIAL_CAPITAL:,}")

    price_df = load_price(symbol)
    sentiment_df = load_sentiment_s3(symbol)

    cash = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    bought_levels = None

    all_trades = []
    window_summary = []

    for window in WINDOWS:
        window_name = window['name']
        train_start, train_end = window['train']
        test_start, test_end = window['test']

        train_df = prepare_data(price_df, sentiment_df, train_start, train_end)
        test_df = prepare_data(price_df, sentiment_df, test_start, test_end)

        if len(train_df) < 100 or len(test_df) < 10:
            continue

        # 训练期网格搜索
        best_params, train_return = grid_search(train_df)
        and_t, or_t = best_params

        # 期初状态
        start_price = test_df['Close'].iloc[0]
        start_total = cash + position * start_price

        print(f"\n\n{'═'*100}")
        print(f"  {window_name}: 测试期 {test_start} ~ {test_end}")
        print(f"  训练期最优参数: AND>{and_t}, OR>{or_t} (训练收益: +{train_return:.1f}%)")
        print(f"  期初状态: 现金=${cash:,.2f}, 持仓={position}股 × ${start_price:.2f} = ${position*start_price:,.2f}")
        print(f"  期初总资产: ${start_total:,.2f}")
        print(f"{'═'*100}")

        # 测试期详细回测
        final_value, trades, cash, position, entry_price, bought_levels = run_threshold_staged_verbose(
            test_df, and_t, or_t, cash, position, entry_price, bought_levels, window_name
        )

        # 计算收益
        test_return = (final_value / start_total - 1) * 100

        window_summary.append({
            'window': window_name,
            'start_total': start_total,
            'end_total': final_value,
            'test_return': test_return,
            'num_trades': len(trades)
        })

        all_trades.extend(trades)

        print(f"\n  {window_name} 收益: ${start_total:,.2f} → ${final_value:,.2f} ({test_return:+.1f}%)")

    # 最终汇总
    print(f"\n\n{'═'*100}")
    print(f"  最终汇总")
    print(f"{'═'*100}")

    print(f"\n  {'窗口':<8} {'期初资产':>15} {'期末资产':>15} {'收益率':>12} {'交易次数':>10}")
    print(f"  {'-'*65}")
    for ws in window_summary:
        print(f"  {ws['window']:<8} ${ws['start_total']:>13,.0f} ${ws['end_total']:>13,.0f} {ws['test_return']:>+10.1f}% {ws['num_trades']:>10}")

    print(f"  {'-'*65}")
    print(f"  {'总计':<8} ${INITIAL_CAPITAL:>13,} ${final_value:>13,.0f} {(final_value/INITIAL_CAPITAL-1)*100:>+10.1f}% {len(all_trades):>10}")


if __name__ == "__main__":
    main()
