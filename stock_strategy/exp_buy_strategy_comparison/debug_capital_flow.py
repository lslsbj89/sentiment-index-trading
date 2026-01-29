"""
调试脚本: 检查每个测试期的资金变化
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


def run_threshold_staged(df, and_threshold, or_threshold,
                         cash, position, entry_price, bought_levels=None):
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

            if current_sentiment > or_threshold:
                sell_signal = True
                sell_reason = f"OR: {current_sentiment:.1f}>{or_threshold}"
            elif current_sentiment > and_threshold and current_price < current_ma50:
                sell_signal = True
                sell_reason = f"AND: {current_sentiment:.1f}>{and_threshold}"

            if sell_signal:
                sell_price = current_price * (1 - SLIPPAGE) * (1 - COMMISSION)
                profit_pct = (sell_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
                sell_value = position * sell_price
                cash += sell_value

                trades.append({
                    'type': 'SELL',
                    'date': current_date,
                    'price': current_price,
                    'shares': position,
                    'sentiment': current_sentiment,
                    'reason': sell_reason,
                    'profit_pct': profit_pct
                })

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

                    trades.append({
                        'type': 'BUY',
                        'date': current_date,
                        'price': current_price,
                        'shares': shares,
                        'sentiment': current_sentiment,
                        'reason': f"Batch{level_idx+1}: sent {current_sentiment:.1f}<{threshold}",
                        'batch': level_idx + 1
                    })

    final_value = cash + position * df['Close'].iloc[-1] if len(df) > 0 else cash
    return final_value, trades, cash, position, entry_price, bought_levels


def grid_search(train_df):
    best_return = -float('inf')
    best_params = None

    for and_t in AND_SELL_RANGE:
        for or_t in OR_SELL_RANGE:
            final_value, _, _, _, _, _ = run_threshold_staged(
                train_df, and_t, or_t, INITIAL_CAPITAL, 0, 0, None
            )
            ret = (final_value / INITIAL_CAPITAL - 1) * 100

            if ret > best_return:
                best_return = ret
                best_params = (and_t, or_t)

    return best_params, best_return


def main():
    symbol = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"

    print("=" * 80)
    print(f"  {symbol} - 资金流动详细检查")
    print("=" * 80)

    price_df = load_price(symbol)
    sentiment_df = load_sentiment_s3(symbol)

    # 连续回测状态
    cash = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    bought_levels = None

    print(f"\n{'窗口':<8} {'测试期初':>18} {'测试期末':>18} {'收益':>12} {'计算公式':<40}")
    print(f"{'':8} {'现金':>9} {'股票':>8} {'现金':>9} {'股票':>8}")
    print("-" * 100)

    capital_flow = []

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

        # 测试期开始时的状态
        start_price = test_df['Close'].iloc[0]
        start_cash = cash
        start_position = position
        start_stock_value = position * start_price
        start_total = start_cash + start_stock_value

        # 测试期回测
        final_value, trades, cash, position, entry_price, bought_levels = run_threshold_staged(
            test_df, and_t, or_t, cash, position, entry_price, bought_levels
        )

        # 测试期结束时的状态
        end_price = test_df['Close'].iloc[-1]
        end_cash = cash
        end_position = position
        end_stock_value = position * end_price
        end_total = end_cash + end_stock_value

        # 计算收益
        test_return = (end_total / start_total - 1) * 100 if start_total > 0 else 0

        # 记录详细信息
        capital_flow.append({
            'window': window_name,
            'test_year': test_start[:4],
            'and_t': and_t,
            'or_t': or_t,
            'start_cash': start_cash,
            'start_position': start_position,
            'start_price': start_price,
            'start_stock_value': start_stock_value,
            'start_total': start_total,
            'end_cash': end_cash,
            'end_position': end_position,
            'end_price': end_price,
            'end_stock_value': end_stock_value,
            'end_total': end_total,
            'test_return': test_return,
            'num_buys': len([t for t in trades if t['type'] == 'BUY']),
            'num_sells': len([t for t in trades if t['type'] == 'SELL']),
        })

        # 打印简要信息
        print(f"{window_name:<8} ${start_cash:>8,.0f} ${start_stock_value:>7,.0f} ${end_cash:>8,.0f} ${end_stock_value:>7,.0f} {test_return:>+10.2f}%  ({end_total:.0f}/{start_total:.0f}-1)*100")

    print("-" * 100)
    print(f"\n总收益: ${INITIAL_CAPITAL:,} → ${end_total:,.0f} ({(end_total/INITIAL_CAPITAL-1)*100:+.1f}%)")

    # 详细表格
    print("\n" + "=" * 80)
    print("  详细资金流动表")
    print("=" * 80)

    for cf in capital_flow:
        print(f"\n{cf['window']} (测试期: {cf['test_year']}, 参数: AND>{cf['and_t']}, OR>{cf['or_t']})")
        print(f"  期初: 现金=${cf['start_cash']:,.2f}, 持仓={cf['start_position']}股 × ${cf['start_price']:.2f} = ${cf['start_stock_value']:,.2f}")
        print(f"        总资产 = ${cf['start_total']:,.2f}")
        print(f"  期末: 现金=${cf['end_cash']:,.2f}, 持仓={cf['end_position']}股 × ${cf['end_price']:.2f} = ${cf['end_stock_value']:,.2f}")
        print(f"        总资产 = ${cf['end_total']:,.2f}")
        print(f"  收益: ({cf['end_total']:,.2f} / {cf['start_total']:,.2f} - 1) × 100 = {cf['test_return']:+.2f}%")
        print(f"  交易: 买入{cf['num_buys']}次, 卖出{cf['num_sells']}次")

    # 验证连续性
    print("\n" + "=" * 80)
    print("  连续性验证")
    print("=" * 80)

    for i in range(1, len(capital_flow)):
        prev = capital_flow[i-1]
        curr = capital_flow[i]

        prev_end_total = prev['end_total']
        curr_start_total = curr['start_total']

        # 如果有持仓，期末和期初价格不同会导致总资产不同
        diff = curr_start_total - prev_end_total
        diff_pct = (curr_start_total / prev_end_total - 1) * 100 if prev_end_total > 0 else 0

        print(f"\n{prev['window']} → {curr['window']}:")
        print(f"  {prev['window']}期末: ${prev_end_total:,.2f} (持仓{prev['end_position']}股 × ${prev['end_price']:.2f})")
        print(f"  {curr['window']}期初: ${curr_start_total:,.2f} (持仓{curr['start_position']}股 × ${curr['start_price']:.2f})")

        if prev['end_position'] > 0:
            price_change = curr['start_price'] - prev['end_price']
            print(f"  价格变化: ${prev['end_price']:.2f} → ${curr['start_price']:.2f} ({price_change:+.2f})")
            print(f"  因持仓导致的资产变化: ${diff:,.2f} ({diff_pct:+.2f}%)")
        else:
            print(f"  无持仓，资金应完全相同: 差异=${diff:.2f}")


if __name__ == "__main__":
    main()
