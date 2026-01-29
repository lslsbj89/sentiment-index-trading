"""
买入策略对比可视化: 阈值分批 vs 间隔分批
==========================================
同一情绪指数下, 对比两种买入策略的交易过程

v2.0 阈值分批: sent<5/0/-5/-10, 每档25%
v3.0 间隔分批: sent<0, 每7天买20%, 回升全买

用法:
  python visualize_buy_strategy_comparison.py SYMBOL [TABLE]
  python visualize_buy_strategy_comparison.py NVDA         # 默认 s3
  python visualize_buy_strategy_comparison.py TSLA s5
"""

import sys
sys.path.insert(0, '/Users/sc2025/Desktop/Claude/Quantrade/src')

import pandas as pd
import numpy as np
import psycopg2
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
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

# v2.0 阈值分批
BUY_THRESHOLDS = [5, 0, -5, -10]
BATCH_PCT = 0.25

# v3.0 间隔分批
INTERVAL_BUY_THRESHOLD = 0
INTERVAL_DAYS = 7
INTERVAL_BATCH_PCT = 0.20

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

TABLE_ALIASES = {
    "s3":     ("fear_greed_index_s3",     "S3"),
    "s5":     ("fear_greed_index_s5",     "S5"),
    "mf26":   ("fear_greed_index",        "MF26"),
    "s3_vix": ("fear_greed_index_s3_vix", "S3+VIX"),
    "s5_vix": ("fear_greed_index_s5_vix", "S5+VIX"),
}

STYLE_STAGED   = {"line": "#2255aa", "buy_marker": "limegreen", "buy_edge": "darkgreen"}
STYLE_INTERVAL = {"line": "#aa5522", "buy_marker": "cyan",      "buy_edge": "darkcyan"}


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
# 训练期网格搜索 (两种策略共用, 阈值分批)
# ============================================================

def run_train_grid(df, and_t, or_t):
    cash = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    bought_levels = set()
    base = cash

    for i in range(len(df)):
        p = df['Close'].iloc[i]
        s = df['sentiment'].iloc[i]
        ma = df['MA50'].iloc[i]

        if position > 0:
            if s > or_t or (s > and_t and p < ma):
                cash += position * p * (1 - SLIPPAGE) * (1 - COMMISSION)
                position = 0; entry_price = 0; bought_levels = set(); base = cash

        for idx, th in enumerate(BUY_THRESHOLDS):
            if idx not in bought_levels and s < th:
                bp = p * (1 + SLIPPAGE) * (1 + COMMISSION)
                shares = int(base * BATCH_PCT / bp)
                if shares > 0 and cash >= shares * bp:
                    cost = shares * bp
                    entry_price = (entry_price * position + cost) / (position + shares) if position > 0 else bp
                    position += shares; cash -= cost; bought_levels.add(idx)

    return cash + position * df['Close'].iloc[-1] if len(df) > 0 else cash


def grid_search(train_df):
    best_ret = -float('inf')
    best = None
    for a in AND_SELL_RANGE:
        for o in OR_SELL_RANGE:
            fv = run_train_grid(train_df, a, o)
            ret = (fv / INITIAL_CAPITAL - 1) * 100
            if ret > best_ret:
                best_ret = ret; best = (a, o)
    return best, best_ret


# ============================================================
# 测试期: v2.0 阈值分批
# ============================================================

def run_staged(df, and_t, or_t, cash, position, entry_price, bought_levels=None):
    if bought_levels is None:
        bought_levels = set()
    trades = []
    base = cash + position * df['Close'].iloc[0] if len(df) > 0 and position > 0 else cash

    for i in range(len(df)):
        dt = df.index[i]; p = df['Close'].iloc[i]; s = df['sentiment'].iloc[i]; ma = df['MA50'].iloc[i]

        if position > 0:
            sell = False; reason = ""
            if s > or_t: sell = True; reason = f"OR: {s:.0f}>{or_t}"
            elif s > and_t and p < ma: sell = True; reason = f"AND: {s:.0f}>{and_t}&P<MA50"
            if sell:
                sp = p * (1 - SLIPPAGE) * (1 - COMMISSION)
                pnl = (sp - entry_price) / entry_price * 100 if entry_price > 0 else 0
                cash += position * sp
                trades.append({'type': 'SELL', 'date': dt, 'price': p, 'shares': position, 'profit_pct': pnl, 'reason': reason})
                position = 0; entry_price = 0; bought_levels = set(); base = cash

        for idx, th in enumerate(BUY_THRESHOLDS):
            if idx not in bought_levels and s < th:
                bp = p * (1 + SLIPPAGE) * (1 + COMMISSION)
                shares = int(base * BATCH_PCT / bp)
                if shares > 0 and cash >= shares * bp:
                    cost = shares * bp
                    entry_price = (entry_price * position + cost) / (position + shares) if position > 0 else bp
                    position += shares; cash -= cost; bought_levels.add(idx)
                    trades.append({'type': 'BUY', 'date': dt, 'price': p, 'shares': shares, 'reason': f"L{idx+1}: {s:.0f}<{th}"})

    fv = cash + position * df['Close'].iloc[-1] if len(df) > 0 else cash
    return fv, trades, cash, position, entry_price, bought_levels


# ============================================================
# 测试期: v3.0 间隔分批
# ============================================================

def run_interval(df, and_t, or_t, cash, position, entry_price,
                 in_buy_mode=False, last_buy_date=None, buy_base=None, batches=0):
    trades = []

    for i in range(len(df)):
        dt = df.index[i]; p = df['Close'].iloc[i]; s = df['sentiment'].iloc[i]; ma = df['MA50'].iloc[i]

        if position > 0:
            sell = False; reason = ""
            if s > or_t: sell = True; reason = f"OR: {s:.0f}>{or_t}"
            elif s > and_t and p < ma: sell = True; reason = f"AND: {s:.0f}>{and_t}&P<MA50"
            if sell:
                sp = p * (1 - SLIPPAGE) * (1 - COMMISSION)
                pnl = (sp - entry_price) / entry_price * 100 if entry_price > 0 else 0
                cash += position * sp
                trades.append({'type': 'SELL', 'date': dt, 'price': p, 'shares': position, 'profit_pct': pnl, 'reason': reason})
                position = 0; entry_price = 0
                in_buy_mode = False; last_buy_date = None; buy_base = None; batches = 0

        if position == 0 or in_buy_mode:
            if not in_buy_mode and s < INTERVAL_BUY_THRESHOLD:
                in_buy_mode = True; buy_base = cash + position * p; batches = 0; last_buy_date = None

            if in_buy_mode:
                should_buy = False; buy_all = False; reason = ""
                if last_buy_date is None:
                    should_buy = True; reason = f"B1: {s:.0f}<{INTERVAL_BUY_THRESHOLD}"
                elif s >= INTERVAL_BUY_THRESHOLD:
                    should_buy = True; buy_all = True; reason = f"ALL: {s:.0f}>={INTERVAL_BUY_THRESHOLD}"
                elif (dt - last_buy_date).days >= INTERVAL_DAYS:
                    batches += 1; should_buy = True; reason = f"B{batches+1}: +{(dt - last_buy_date).days}d"

                if should_buy:
                    bp = p * (1 + SLIPPAGE) * (1 + COMMISSION)
                    shares = int(cash / bp) if buy_all else int(buy_base * INTERVAL_BATCH_PCT / bp)
                    if shares > 0 and cash >= shares * bp:
                        cost = shares * bp
                        entry_price = (entry_price * position + cost) / (position + shares) if position > 0 else bp
                        position += shares; cash -= cost; last_buy_date = dt
                        trades.append({'type': 'BUY', 'date': dt, 'price': p, 'shares': shares, 'reason': reason})
                    if buy_all:
                        in_buy_mode = False; buy_base = None; batches = 0

    fv = cash + position * df['Close'].iloc[-1] if len(df) > 0 else cash
    return fv, trades, cash, position, entry_price, in_buy_mode, last_buy_date, buy_base, batches


# ============================================================
# Walk-Forward
# ============================================================

def wf_staged(symbol, table):
    price_df = load_price(symbol)
    sentiment_df = load_sentiment(table, symbol)
    cash = INITIAL_CAPITAL; pos = 0; ep = 0; bl = None
    all_trades = []; params = []

    for w in WINDOWS:
        train_df = prepare_data(price_df, sentiment_df, w['train'][0], w['train'][1])
        test_df = prepare_data(price_df, sentiment_df, w['test'][0], w['test'][1])
        if len(train_df) < 100 or len(test_df) < 10: continue
        best, _ = grid_search(train_df)
        start_val = cash + pos * test_df['Close'].iloc[0] if len(test_df) > 0 else cash
        fv, trades, cash, pos, ep, bl = run_staged(test_df, best[0], best[1], cash, pos, ep, bl)
        ret = (fv / start_val - 1) * 100 if start_val > 0 else 0
        params.append({'window': w['name'], 'and': best[0], 'or': best[1], 'return': ret, 'end_value': fv})
        all_trades.extend(trades)

    total_ret = (fv / INITIAL_CAPITAL - 1) * 100
    return all_trades, params, total_ret, fv


def wf_interval(symbol, table):
    price_df = load_price(symbol)
    sentiment_df = load_sentiment(table, symbol)
    cash = INITIAL_CAPITAL; pos = 0; ep = 0
    ibm = False; lbd = None; bb = None; bat = 0
    all_trades = []; params = []

    for w in WINDOWS:
        train_df = prepare_data(price_df, sentiment_df, w['train'][0], w['train'][1])
        test_df = prepare_data(price_df, sentiment_df, w['test'][0], w['test'][1])
        if len(train_df) < 100 or len(test_df) < 10: continue
        best, _ = grid_search(train_df)
        start_val = cash + pos * test_df['Close'].iloc[0] if len(test_df) > 0 else cash
        fv, trades, cash, pos, ep, ibm, lbd, bb, bat = run_interval(
            test_df, best[0], best[1], cash, pos, ep, ibm, lbd, bb, bat)
        ret = (fv / start_val - 1) * 100 if start_val > 0 else 0
        params.append({'window': w['name'], 'and': best[0], 'or': best[1], 'return': ret, 'end_value': fv})
        all_trades.extend(trades)

    total_ret = (fv / INITIAL_CAPITAL - 1) * 100
    return all_trades, params, total_ret, fv


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
        c = 'red' if t.get('profit_pct', 0) > 0 else 'orange'
        e = 'darkred' if t.get('profit_pct', 0) > 0 else 'darkorange'
        ax.scatter(t['date'], t['price'], color=c, marker='v', s=120, edgecolors=e, linewidths=1.2, zorder=5)
    y_min, y_max = ax.get_ylim()
    for p in params:
        ws = [w for w in WINDOWS if w['name'] == p['window']][0]['test'][0]
        ax.text(pd.Timestamp(ws, tz='UTC') + pd.Timedelta(days=5), y_min + (y_max - y_min) * 0.02,
                f"AND>{p['and']} OR>{p['or']}\n{p['return']:+.1f}%", fontsize=7.5, color=style['line'],
                va='bottom', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    ax.set_ylabel('Price ($)', fontsize=11)
    ax.set_title(f'{label} — Total: +{total_ret:.1f}% (${final_val:,.0f})', fontsize=13, color=style['line'], loc='left')
    legend_elements = [
        Line2D([0], [0], color='#333333', linewidth=1.2, label='Price'),
        Line2D([0], [0], color='orange', linewidth=1, linestyle='--', alpha=0.6, label='MA50'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=style['buy_marker'], markersize=10,
               markeredgecolor=style['buy_edge'], markeredgewidth=1.2, label=f'Buy ({len(buys)})', linestyle='None'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='red', markersize=10,
               markeredgecolor='darkred', markeredgewidth=1.2, label=f'Sell ({len(sells)})', linestyle='None'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.2)


def plot_sentiment_row(ax, df, trades, params, style, buy_thresh_lines):
    ax.plot(df.index, df['sentiment'], color=style['line'], linewidth=1, alpha=0.8)
    for val, ls, alpha in buy_thresh_lines:
        ax.axhline(y=val, color='green', linewidth=0.8, linestyle=ls, alpha=alpha)
    fill_y = buy_thresh_lines[0][0]
    ax.fill_between(df.index, df['sentiment'], fill_y, where=df['sentiment'] < fill_y, alpha=0.15, color='green')
    for p in params:
        w = [w for w in WINDOWS if w['name'] == p['window']][0]
        ws = pd.Timestamp(w['test'][0], tz='UTC'); we = pd.Timestamp(w['test'][1], tz='UTC')
        ax.hlines(y=p['or'], xmin=ws, xmax=we, color='red', linewidth=1.2, linestyle='-', alpha=0.6)
        ax.hlines(y=p['and'], xmin=ws, xmax=we, color='red', linewidth=0.8, linestyle='--', alpha=0.4)
        ax.text(ws + pd.Timedelta(days=3), p['or'] + 2, f"OR>{p['or']}", fontsize=7, color='red', alpha=0.7)
        ax.text(ws + pd.Timedelta(days=3), p['and'] - 4, f"AND>{p['and']}", fontsize=7, color='red', alpha=0.5)
    for t in [t for t in trades if t['type'] == 'BUY']:
        ax.axvline(x=t['date'], color='green', linewidth=0.5, alpha=0.3)
    for t in [t for t in trades if t['type'] == 'SELL']:
        ax.axvline(x=t['date'], color='red', linewidth=0.5, alpha=0.3)
    ax.set_ylabel('Sentiment', fontsize=11)
    ax.set_ylim(-50, 80)
    ax.grid(True, alpha=0.2)


def run_comparison(symbol, table_key):
    table_name, display = resolve_table(table_key)

    print(f"Buy Strategy Comparison: Staged vs Interval")
    print(f"Symbol: {symbol} | Index: {display} ({table_name})")

    price_df = load_price(symbol)
    sentiment_df = load_sentiment(table_name, symbol)
    df = prepare_data(price_df, sentiment_df, '2020-01-01', '2025-12-31')

    print("Running v2.0 Staged Buy...")
    trades_s, params_s, ret_s, fv_s = wf_staged(symbol, table_name)

    print("Running v3.0 Interval Buy...")
    trades_i, params_i, ret_i, fv_i = wf_interval(symbol, table_name)

    for tag, params, ret, fv in [("v2.0 Staged", params_s, ret_s, fv_s),
                                  ("v3.0 Interval", params_i, ret_i, fv_i)]:
        print(f"\n{tag}:")
        for p in params:
            print(f"  {p['window']}: AND>{p['and']}, OR>{p['or']} -> {p['return']:+.1f}%")
        print(f"  Total: +{ret:.1f}% (${fv:,.0f})")

    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(28, 20), sharex=True,
                             gridspec_kw={'height_ratios': [3, 2, 3, 2]})
    fig.suptitle(f'{symbol} Buy Strategy: Staged (sent<5/0/-5/-10) vs Interval (sent<0, 7d, 20%)  [{display}]',
                 fontsize=17, fontweight='bold', y=0.98)

    wb = [pd.Timestamp(f'{y}-01-01', tz='UTC') for y in range(2020, 2026)] + [pd.Timestamp('2025-12-31', tz='UTC')]
    wc = ['#f0f8ff', '#fff8f0', '#f8f0f0', '#f0fff0', '#f8f0ff', '#fffff0']
    wl = ['W2020', 'W2021', 'W2022', 'W2023', 'W2024', 'W2025']
    for ax in axes:
        for i in range(len(wb) - 1):
            ax.axvspan(wb[i], wb[i+1], alpha=0.3, color=wc[i], zorder=0)

    # Row 1+2: Staged
    staged_thresh = [(5, '--', 0.5), (0, '-.', 0.4), (-5, ':', 0.4), (-10, ':', 0.3)]
    plot_price_row(axes[0], df, trades_s, params_s, STYLE_STAGED,
                   f'v2.0 Staged Buy (sent<5/0/-5/-10, 4x25%)', ret_s, fv_s)
    plot_sentiment_row(axes[1], df, trades_s, params_s, STYLE_STAGED, staged_thresh)

    # Row 3+4: Interval
    interval_thresh = [(INTERVAL_BUY_THRESHOLD, '--', 0.6)]
    plot_price_row(axes[2], df, trades_i, params_i, STYLE_INTERVAL,
                   f'v3.0 Interval Buy (sent<{INTERVAL_BUY_THRESHOLD}, 7d, 5x20%)', ret_i, fv_i)
    plot_sentiment_row(axes[3], df, trades_i, params_i, STYLE_INTERVAL, interval_thresh)

    for i in range(len(wb) - 1):
        mid = wb[i] + (wb[i+1] - wb[i]) / 2
        axes[0].annotate(wl[i], xy=(mid, 1.02), xycoords=('data', 'axes fraction'),
                        ha='center', fontsize=10, color='gray', fontweight='bold')

    axes[3].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(axes[3].xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = f'{symbol}_staged_vs_interval_{table_key}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out}")

    # Stats
    buys_s = len([t for t in trades_s if t['type'] == 'BUY'])
    sells_s = len([t for t in trades_s if t['type'] == 'SELL'])
    wins_s = len([t for t in trades_s if t['type'] == 'SELL' and t.get('profit_pct', 0) > 0])
    buys_i = len([t for t in trades_i if t['type'] == 'BUY'])
    sells_i = len([t for t in trades_i if t['type'] == 'SELL'])
    wins_i = len([t for t in trades_i if t['type'] == 'SELL' and t.get('profit_pct', 0) > 0])

    print(f"\n{'='*55}")
    print(f"{symbol} [{display}] Buy Strategy Comparison")
    print(f"{'='*55}")
    print(f"{'':>16} {'v2.0 Staged':>16} {'v3.0 Interval':>16}")
    print(f"{'Total Return':>16} {ret_s:>+15.1f}% {ret_i:>+15.1f}%")
    print(f"{'Final Value':>16} ${fv_s:>14,.0f} ${fv_i:>14,.0f}")
    print(f"{'Buys':>16} {buys_s:>16} {buys_i:>16}")
    print(f"{'Sells':>16} {sells_s:>16} {sells_i:>16}")
    print(f"{'Winning Sells':>16} {wins_s:>16} {wins_i:>16}")

    return ret_s, fv_s, ret_i, fv_i


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_buy_strategy_comparison.py SYMBOL [TABLE]")
        print("  TABLE: s3 (default), s5, mf26, s3_vix, s5_vix")
        sys.exit(1)

    symbol = sys.argv[1].upper()
    table_key = sys.argv[2] if len(sys.argv) > 2 else "s3"

    if symbol == "ALL":
        symbols = ["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        results = []
        for s in symbols:
            ret_s, fv_s, ret_i, fv_i = run_comparison(s, table_key)
            results.append((s, ret_s, fv_s, ret_i, fv_i))

        print(f"\n\n{'='*65}")
        print(f"SUMMARY: Staged vs Interval [{table_key.upper()}]")
        print(f"{'='*65}")
        print(f"{'Stock':<8} {'v2.0 Staged':>14} {'v3.0 Interval':>14} {'Diff':>10} {'Winner':>10}")
        print(f"{'-'*56}")
        total_s = total_i = 0
        staged_wins = interval_wins = 0
        for s, rs, fs, ri, fi in results:
            diff = ri - rs
            winner = "Interval" if diff > 0 else "Staged"
            if diff > 0: interval_wins += 1
            else: staged_wins += 1
            print(f"{s:<8} {rs:>+13.1f}% {ri:>+13.1f}% {diff:>+9.1f}% {winner:>10}")
            total_s += fs; total_i += fi
        print(f"{'-'*56}")
        print(f"{'Total $':<8} ${total_s:>12,.0f} ${total_i:>12,.0f}")
        print(f"{'Score':<8} {'':>14} {'':>14} {'':>10} {staged_wins}:{interval_wins}")
    else:
        run_comparison(symbol, table_key)
