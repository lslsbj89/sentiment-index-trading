"""
情绪指数对比可视化 (间隔买入版 v3.0)
====================================
使用间隔分批买入策略对比任意两种情绪指数

买入策略 (测试期):
  情绪 < 0 → 进入买入模式, 买20%
  每隔7天 → 再买20%
  情绪 >= 0 → 买入全部剩余, 退出买入模式

卖出策略: OR/AND 双条件 (训练期网格搜索)

4行布局:
  Row 1: 价格 + 方法A 买卖信号
  Row 2: 方法A 情绪指数 + 阈值线
  Row 3: 价格 + 方法B 买卖信号
  Row 4: 方法B 情绪指数 + 阈值线

用法:
  python visualize_interval_buy_comparison.py SYMBOL TABLE_A TABLE_B

  TABLE 可用简称:
    s3       → fear_greed_index_s3       (Smoothing=3, MF=13)
    s5       → fear_greed_index_s5       (Smoothing=5)
    mf26     → fear_greed_index          (Smoothing=5, MF=26)
    s3_vix   → fear_greed_index_s3_vix   (S3 + VIX四因子)
    s5_vix   → fear_greed_index_s5_vix   (S5 + VIX四因子)

示例:
  python visualize_interval_buy_comparison.py TSLA s3 mf26
  python visualize_interval_buy_comparison.py NVDA s3 s3_vix
"""

import sys
sys.path.insert(0, '/Users/sc2025/Desktop/Claude/Quantrade/src')

import pandas as pd
import numpy as np
import psycopg2
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from datetime import datetime
from data_loader import DataLoader

# ============================================================
# 配置
# ============================================================

DB_CONFIG = {
    "host": "localhost", "port": 5432,
    "database": "crypto_fear_greed_2",
    "user": "sc2025", "password": ""
}

INITIAL_CAPITAL = 100000
COMMISSION = 0.001
SLIPPAGE = 0.001

# 训练期: 原版阈值分批 (用于网格搜索卖出参数)
BUY_THRESHOLDS = [5, 0, -5, -10]
BATCH_PCT = 0.25

# 测试期: 间隔分批买入 (v3.0 生产版)
INTERVAL_BUY_THRESHOLD = 0      # 买入触发: 情绪 < 0
INTERVAL_DAYS = 7               # 间隔天数
INTERVAL_BATCH_PCT = 0.20       # 每次买入20%

# 卖出参数搜索范围
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

# 简称 → (完整表名, 显示名称)
TABLE_ALIASES = {
    "s3":     ("fear_greed_index_s3",     "S3 (Smooth=3, MF=13)"),
    "s5":     ("fear_greed_index_s5",     "S5 (Smooth=5)"),
    "mf26":   ("fear_greed_index",        "MF26 (Smooth=5, MF=26)"),
    "s3_vix": ("fear_greed_index_s3_vix", "S3+VIX"),
    "s5_vix": ("fear_greed_index_s5_vix", "S5+VIX"),
}

STYLE_A = {"line": "#2255aa", "buy_marker": "limegreen", "buy_edge": "darkgreen"}
STYLE_B = {"line": "#aa5522", "buy_marker": "cyan",      "buy_edge": "darkcyan"}


def resolve_table(name):
    key = name.lower().strip()
    if key in TABLE_ALIASES:
        return TABLE_ALIASES[key]
    return (name, name)


# ============================================================
# 数据加载
# ============================================================

def load_sentiment(table, symbol):
    conn = psycopg2.connect(**DB_CONFIG)
    query = f"SELECT date, smoothed_index FROM {table} WHERE symbol = %s ORDER BY date"
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
    return df[start_date:end_date]


# ============================================================
# 训练期回测: 阈值分批 (用于网格搜索卖出参数)
# ============================================================

def run_threshold_staged_train(df, and_threshold, or_threshold):
    cash = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    bought_levels = set()
    initial_capital_for_batch = cash

    for i in range(len(df)):
        price = df['Close'].iloc[i]
        sent = df['sentiment'].iloc[i]
        ma50 = df['MA50'].iloc[i]

        if position > 0:
            if sent > or_threshold or (sent > and_threshold and price < ma50):
                cash += position * price * (1 - SLIPPAGE) * (1 - COMMISSION)
                position = 0
                entry_price = 0
                bought_levels = set()
                initial_capital_for_batch = cash

        for idx, th in enumerate(BUY_THRESHOLDS):
            if idx not in bought_levels and sent < th:
                bp = price * (1 + SLIPPAGE) * (1 + COMMISSION)
                shares = int(initial_capital_for_batch * BATCH_PCT / bp)
                if shares > 0 and cash >= shares * bp:
                    cost = shares * bp
                    if position > 0:
                        entry_price = (entry_price * position + cost) / (position + shares)
                        position += shares
                    else:
                        position = shares
                        entry_price = bp
                    cash -= cost
                    bought_levels.add(idx)

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
# 测试期回测: 间隔分批买入 (v3.0)
# ============================================================

def run_interval_buy(df, and_threshold, or_threshold,
                     cash, position, entry_price,
                     in_buy_mode=False, last_buy_date=None,
                     buy_base_capital=None, batches_bought=0):
    trades = []

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
                sell_reason = f"OR: sent {current_sentiment:.1f} > {or_threshold}"
            elif current_sentiment > and_threshold and current_price < current_ma50:
                sell_signal = True
                sell_reason = f"AND: sent {current_sentiment:.1f} > {and_threshold} & P<MA50"

            if sell_signal:
                sell_price = current_price * (1 - SLIPPAGE) * (1 - COMMISSION)
                profit_pct = (sell_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
                cash += position * sell_price
                trades.append({
                    'type': 'SELL', 'date': current_date, 'price': current_price,
                    'shares': position, 'profit_pct': profit_pct, 'reason': sell_reason
                })
                position = 0
                entry_price = 0
                in_buy_mode = False
                last_buy_date = None
                buy_base_capital = None
                batches_bought = 0

        # 买入逻辑: 间隔分批
        if position == 0 or in_buy_mode:
            if not in_buy_mode and current_sentiment < INTERVAL_BUY_THRESHOLD:
                in_buy_mode = True
                buy_base_capital = cash + position * current_price
                batches_bought = 0
                last_buy_date = None

            if in_buy_mode:
                should_buy = False
                buy_reason = ""
                buy_all_remaining = False

                if last_buy_date is None:
                    should_buy = True
                    buy_reason = f"B1: sent {current_sentiment:.1f} < {INTERVAL_BUY_THRESHOLD}"
                elif current_sentiment >= INTERVAL_BUY_THRESHOLD:
                    should_buy = True
                    buy_all_remaining = True
                    buy_reason = f"ALL: sent {current_sentiment:.1f} >= {INTERVAL_BUY_THRESHOLD}"
                elif (current_date - last_buy_date).days >= INTERVAL_DAYS:
                    batches_bought += 1
                    should_buy = True
                    buy_reason = f"B{batches_bought+1}: +{(current_date - last_buy_date).days}d"

                if should_buy:
                    buy_price = current_price * (1 + SLIPPAGE) * (1 + COMMISSION)
                    if buy_all_remaining:
                        shares = int(cash / buy_price)
                    else:
                        shares = int(buy_base_capital * INTERVAL_BATCH_PCT / buy_price)

                    if shares > 0 and cash >= shares * buy_price:
                        buy_cost = shares * buy_price
                        if position > 0:
                            entry_price = (entry_price * position + buy_cost) / (position + shares)
                            position += shares
                        else:
                            position = shares
                            entry_price = buy_price
                        cash -= buy_cost
                        last_buy_date = current_date
                        trades.append({
                            'type': 'BUY', 'date': current_date, 'price': current_price,
                            'shares': shares, 'reason': buy_reason
                        })

                    if buy_all_remaining:
                        in_buy_mode = False
                        buy_base_capital = None
                        batches_bought = 0

    final_value = cash + position * df['Close'].iloc[-1] if len(df) > 0 else cash
    return (final_value, trades, cash, position, entry_price,
            in_buy_mode, last_buy_date, buy_base_capital, batches_bought)


# ============================================================
# Walk-Forward (间隔买入)
# ============================================================

def run_walk_forward_full(symbol, sentiment_table):
    price_df = load_price(symbol)
    sentiment_df = load_sentiment(sentiment_table, symbol)

    cash = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    in_buy_mode = False
    last_buy_date = None
    buy_base_capital = None
    batches_bought = 0

    all_trades = []
    window_params = []

    for window in WINDOWS:
        train_df = prepare_data(price_df, sentiment_df, window['train'][0], window['train'][1])
        test_df = prepare_data(price_df, sentiment_df, window['test'][0], window['test'][1])

        if len(train_df) < 100 or len(test_df) < 10:
            continue

        best_params, _ = grid_search(train_df)
        and_t, or_t = best_params

        test_start_value = cash + position * test_df['Close'].iloc[0] if len(test_df) > 0 else cash

        (final_value, trades, cash, position, entry_price,
         in_buy_mode, last_buy_date, buy_base_capital, batches_bought) = run_interval_buy(
            test_df, and_t, or_t, cash, position, entry_price,
            in_buy_mode, last_buy_date, buy_base_capital, batches_bought
        )

        test_return = (final_value / test_start_value - 1) * 100 if test_start_value > 0 else 0
        window_params.append({
            'window': window['name'],
            'and': and_t, 'or': or_t,
            'return': test_return, 'end_value': final_value
        })
        all_trades.extend(trades)

    total_return = (final_value / INITIAL_CAPITAL - 1) * 100
    return all_trades, window_params, total_return, final_value


# ============================================================
# 绘图
# ============================================================

def plot_price_row(ax, df, trades, params, style, label, total_ret, final_val):
    ax.plot(df.index, df['Close'], color='#333333', linewidth=1.2, alpha=0.9)
    ax.plot(df.index, df['MA50'], color='orange', linewidth=1, alpha=0.6, linestyle='--')

    buys = [t for t in trades if t['type'] == 'BUY']
    sells = [t for t in trades if t['type'] == 'SELL']

    for t in buys:
        ax.scatter(t['date'], t['price'], color=style['buy_marker'], marker='^', s=120,
                   edgecolors=style['buy_edge'], linewidths=1.2, zorder=5)
    for t in sells:
        color = 'red' if t.get('profit_pct', 0) > 0 else 'orange'
        edge = 'darkred' if t.get('profit_pct', 0) > 0 else 'darkorange'
        ax.scatter(t['date'], t['price'], color=color, marker='v', s=120,
                   edgecolors=edge, linewidths=1.2, zorder=5)

    y_min, y_max = ax.get_ylim()
    for p in params:
        w_start = [w for w in WINDOWS if w['name'] == p['window']][0]['test'][0]
        ax.text(pd.Timestamp(w_start, tz='UTC') + pd.Timedelta(days=5),
                y_min + (y_max - y_min) * 0.02,
                f"AND>{p['and']} OR>{p['or']}\n{p['return']:+.1f}%",
                fontsize=7.5, color=style['line'], va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    ax.set_ylabel('Price ($)', fontsize=11)
    ax.set_title(f'{label} — Total: +{total_ret:.1f}% (${final_val:,.0f})',
                 fontsize=13, color=style['line'], loc='left')

    legend_elements = [
        Line2D([0], [0], color='#333333', linewidth=1.2, label='Price'),
        Line2D([0], [0], color='orange', linewidth=1, linestyle='--', alpha=0.6, label='MA50'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=style['buy_marker'],
               markersize=10, markeredgecolor=style['buy_edge'], markeredgewidth=1.2,
               label=f'Buy ({len(buys)})', linestyle='None'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='red',
               markersize=10, markeredgecolor='darkred', markeredgewidth=1.2,
               label=f'Sell ({len(sells)})', linestyle='None'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.2)


def plot_sentiment_row(ax, df, trades, params, style, label):
    ax.plot(df.index, df['sentiment'], color=style['line'], linewidth=1, alpha=0.8)

    # 买入阈值线: 间隔买入用 0
    ax.axhline(y=INTERVAL_BUY_THRESHOLD, color='green', linewidth=1.2,
               linestyle='--', alpha=0.6, label=f'Buy < {INTERVAL_BUY_THRESHOLD}')
    ax.fill_between(df.index, df['sentiment'], INTERVAL_BUY_THRESHOLD,
                     where=df['sentiment'] < INTERVAL_BUY_THRESHOLD, alpha=0.15, color='green')

    for p in params:
        w = [w for w in WINDOWS if w['name'] == p['window']][0]
        ws = pd.Timestamp(w['test'][0], tz='UTC')
        we = pd.Timestamp(w['test'][1], tz='UTC')
        ax.hlines(y=p['or'], xmin=ws, xmax=we, color='red', linewidth=1.2,
                  linestyle='-', alpha=0.6)
        ax.hlines(y=p['and'], xmin=ws, xmax=we, color='red', linewidth=0.8,
                  linestyle='--', alpha=0.4)
        ax.text(ws + pd.Timedelta(days=3), p['or'] + 2,
                f"OR>{p['or']}", fontsize=7, color='red', alpha=0.7)
        ax.text(ws + pd.Timedelta(days=3), p['and'] - 4,
                f"AND>{p['and']}", fontsize=7, color='red', alpha=0.5)

    buys = [t for t in trades if t['type'] == 'BUY']
    sells = [t for t in trades if t['type'] == 'SELL']
    for t in buys:
        ax.axvline(x=t['date'], color='green', linewidth=0.5, alpha=0.3)
    for t in sells:
        ax.axvline(x=t['date'], color='red', linewidth=0.5, alpha=0.3)

    ax.set_ylabel(f'Sentiment ({label})', fontsize=11)
    ax.set_ylim(-50, 80)
    ax.grid(True, alpha=0.2)


def plot_comparison(symbol, table_a, table_b):
    table_name_a, display_a = resolve_table(table_a)
    table_name_b, display_b = resolve_table(table_b)

    print(f"Interval Buy Comparison: {display_a} vs {display_b}")
    print(f"Symbol: {symbol}")
    print(f"Buy: sent < {INTERVAL_BUY_THRESHOLD}, interval {INTERVAL_DAYS}d, batch {INTERVAL_BATCH_PCT*100:.0f}%")

    price_df = load_price(symbol)
    sentiment_a = load_sentiment(table_name_a, symbol)
    sentiment_b = load_sentiment(table_name_b, symbol)

    test_start, test_end = '2020-01-01', '2025-12-31'
    df_a = prepare_data(price_df, sentiment_a, test_start, test_end)
    df_b = prepare_data(price_df, sentiment_b, test_start, test_end)

    print(f"Running {display_a} Walk-Forward...")
    trades_a, params_a, ret_a, fv_a = run_walk_forward_full(symbol, table_name_a)

    print(f"Running {display_b} Walk-Forward...")
    trades_b, params_b, ret_b, fv_b = run_walk_forward_full(symbol, table_name_b)

    for tag, params, ret, fv, disp in [("A", params_a, ret_a, fv_a, display_a),
                                        ("B", params_b, ret_b, fv_b, display_b)]:
        print(f"\n{disp}:")
        for p in params:
            print(f"  {p['window']}: AND>{p['and']}, OR>{p['or']} -> {p['return']:+.1f}%")
        print(f"  Total: +{ret:.1f}% (${fv:,.0f})")

    # ========== Plot ==========
    fig, axes = plt.subplots(4, 1, figsize=(28, 20), sharex=True,
                             gridspec_kw={'height_ratios': [3, 2, 3, 2]})
    fig.suptitle(f'{symbol} Interval Buy Comparison: {display_a} vs {display_b} (2020-2025)',
                 fontsize=18, fontweight='bold', y=0.98)

    window_boundaries = [
        pd.Timestamp('2020-01-01', tz='UTC'),
        pd.Timestamp('2021-01-01', tz='UTC'),
        pd.Timestamp('2022-01-01', tz='UTC'),
        pd.Timestamp('2023-01-01', tz='UTC'),
        pd.Timestamp('2024-01-01', tz='UTC'),
        pd.Timestamp('2025-01-01', tz='UTC'),
        pd.Timestamp('2025-12-31', tz='UTC'),
    ]
    window_colors = ['#f0f8ff', '#fff8f0', '#f8f0f0', '#f0fff0', '#f8f0ff', '#fffff0']
    window_labels = ['W2020', 'W2021', 'W2022', 'W2023', 'W2024', 'W2025']

    for ax in axes:
        for i in range(len(window_boundaries) - 1):
            ax.axvspan(window_boundaries[i], window_boundaries[i+1],
                      alpha=0.3, color=window_colors[i], zorder=0)

    plot_price_row(axes[0], df_a, trades_a, params_a, STYLE_A, display_a, ret_a, fv_a)
    plot_sentiment_row(axes[1], df_a, trades_a, params_a, STYLE_A, display_a)
    plot_price_row(axes[2], df_b, trades_b, params_b, STYLE_B, display_b, ret_b, fv_b)
    plot_sentiment_row(axes[3], df_b, trades_b, params_b, STYLE_B, display_b)

    for i in range(len(window_boundaries) - 1):
        mid = window_boundaries[i] + (window_boundaries[i+1] - window_boundaries[i]) / 2
        axes[0].annotate(window_labels[i], xy=(mid, 1.02), xycoords=('data', 'axes fraction'),
                        ha='center', fontsize=10, color='gray', fontweight='bold')

    axes[3].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(axes[3].xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    short_a = table_a.lower().strip()
    short_b = table_b.lower().strip()
    output_file = f'{symbol}_interval_{short_a}_vs_{short_b}_trades.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_file}")

    # Stats
    buys_a = [t for t in trades_a if t['type'] == 'BUY']
    sells_a = [t for t in trades_a if t['type'] == 'SELL']
    buys_b = [t for t in trades_b if t['type'] == 'BUY']
    sells_b = [t for t in trades_b if t['type'] == 'SELL']
    win_a = len([t for t in sells_a if t.get('profit_pct', 0) > 0])
    win_b = len([t for t in sells_b if t.get('profit_pct', 0) > 0])

    col_a = display_a[:18]
    col_b = display_b[:18]
    print(f"\n{'='*60}")
    print(f"{symbol} Trading Stats (Interval Buy, sent<{INTERVAL_BUY_THRESHOLD}, 7d, 20%)")
    print(f"{'='*60}")
    print(f"{'':>16} {col_a:>18} {col_b:>18}")
    print(f"{'Total Return':>16} {ret_a:>+17.1f}% {ret_b:>+17.1f}%")
    print(f"{'Final Value':>16} ${fv_a:>16,.0f} ${fv_b:>16,.0f}")
    print(f"{'Buys':>16} {len(buys_a):>18} {len(buys_b):>18}")
    print(f"{'Sells':>16} {len(sells_a):>18} {len(sells_b):>18}")
    print(f"{'Winning Sells':>16} {win_a:>18} {win_b:>18}")


def print_usage():
    print("Usage: python visualize_interval_buy_comparison.py SYMBOL TABLE_A TABLE_B")
    print()
    print("Buy strategy: Interval Buy v3.0 (sent<0, 7d interval, 20% batch)")
    print()
    print("Available aliases:")
    for key, (table, display) in TABLE_ALIASES.items():
        print(f"  {key:<10} -> {table:<30} {display}")
    print()
    print("Examples:")
    print("  python visualize_interval_buy_comparison.py NVDA s3 s3_vix")
    print("  python visualize_interval_buy_comparison.py TSLA s3 s5")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        if len(sys.argv) == 2 and sys.argv[1] in ('-h', '--help'):
            print_usage()
            sys.exit(0)
        print("Missing arguments!")
        print_usage()
        sys.exit(1)

    symbol = sys.argv[1].upper()
    table_a = sys.argv[2]
    table_b = sys.argv[3]

    plot_comparison(symbol, table_a, table_b)
