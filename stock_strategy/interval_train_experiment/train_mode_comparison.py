"""
Training Mode Comparison: Staged vs Interval Buy Training
=========================================================
Compare two approaches for grid-searching sell parameters:

A) Staged train (current):   train grid search uses staged buy [5,0,-5,-10]
                              test uses interval buy (sent<0, 7d, 20%)
B) Interval train (proposed): train grid search uses interval buy (sent<0, 7d, 20%)
                              test uses interval buy (sent<0, 7d, 20%)

Hypothesis: matching train/test buy strategy should improve results,
            as proven in crypto (+9.8% improvement).

Usage:
  python train_mode_comparison.py NVDA        # Single stock
  python train_mode_comparison.py ALL         # All 7 stocks
  python train_mode_comparison.py ALL s5      # With different sentiment table
"""

import sys
sys.path.insert(0, '/Users/sc2025/Desktop/Claude/Quantrade/src')

import pandas as pd
import numpy as np
import psycopg2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_loader import DataLoader

# ============================================================
# Configuration
# ============================================================

DB_CONFIG = {
    "host": "localhost", "port": 5432,
    "database": "crypto_fear_greed_2",
    "user": "sc2025", "password": ""
}

INITIAL_CAPITAL = 100000
COMMISSION = 0.001
SLIPPAGE = 0.001

# v2.0 Staged buy (training mode A)
BUY_THRESHOLDS = [5, 0, -5, -10]
BATCH_PCT = 0.25

# v3.0 Interval buy (training mode B + test mode for both)
INTERVAL_BUY_THRESHOLD = 0
INTERVAL_DAYS = 7
INTERVAL_BATCH_PCT = 0.20

# Sell parameter search grid
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

TABLE_ALIASES = {
    "s3":     ("fear_greed_index_s3",     "S3"),
    "s5":     ("fear_greed_index_s5",     "S5"),
    "mf26":   ("fear_greed_index",        "MF26"),
    "s3_vix": ("fear_greed_index_s3_vix", "S3+VIX"),
    "s5_vix": ("fear_greed_index_s5_vix", "S5+VIX"),
}


def resolve_table(name):
    key = name.lower().strip()
    if key in TABLE_ALIASES:
        return TABLE_ALIASES[key]
    return (name, name)


# ============================================================
# Data Loading
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
# Training Mode A: Staged buy grid search (current approach)
# ============================================================

def run_train_staged(df, and_t, or_t):
    """Grid search training using staged buy [5,0,-5,-10], 4x25%."""
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
                position = 0
                entry_price = 0
                bought_levels = set()
                base = cash

        for idx, th in enumerate(BUY_THRESHOLDS):
            if idx not in bought_levels and s < th:
                bp = p * (1 + SLIPPAGE) * (1 + COMMISSION)
                shares = int(base * BATCH_PCT / bp)
                if shares > 0 and cash >= shares * bp:
                    cost = shares * bp
                    entry_price = (entry_price * position + cost) / (position + shares) if position > 0 else bp
                    position += shares
                    cash -= cost
                    bought_levels.add(idx)

    return cash + position * df['Close'].iloc[-1] if len(df) > 0 else cash


def grid_search_staged(train_df):
    """Grid search using staged buy training."""
    best_ret = -float('inf')
    best = None
    for a in AND_SELL_RANGE:
        for o in OR_SELL_RANGE:
            fv = run_train_staged(train_df, a, o)
            ret = (fv / INITIAL_CAPITAL - 1) * 100
            if ret > best_ret:
                best_ret = ret
                best = (a, o)
    return best, best_ret


# ============================================================
# Training Mode B: Interval buy grid search (proposed)
# ============================================================

def run_train_interval(df, and_t, or_t):
    """Grid search training using interval buy (sent<0, 7d, 20%). Matches test strategy."""
    cash = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    in_buy_mode = False
    last_buy_date = None
    buy_base = None
    batches = 0

    for i in range(len(df)):
        dt = df.index[i]
        p = df['Close'].iloc[i]
        s = df['sentiment'].iloc[i]
        ma = df['MA50'].iloc[i]

        # Sell
        if position > 0:
            if s > or_t or (s > and_t and p < ma):
                cash += position * p * (1 - SLIPPAGE) * (1 - COMMISSION)
                position = 0
                entry_price = 0
                in_buy_mode = False
                last_buy_date = None
                buy_base = None
                batches = 0

        # Buy (interval logic)
        if position == 0 or in_buy_mode:
            if not in_buy_mode and s < INTERVAL_BUY_THRESHOLD:
                in_buy_mode = True
                buy_base = cash + position * p
                batches = 0
                last_buy_date = None

            if in_buy_mode:
                should_buy = False
                buy_all = False
                if last_buy_date is None:
                    should_buy = True
                elif s >= INTERVAL_BUY_THRESHOLD:
                    should_buy = True
                    buy_all = True
                elif (dt - last_buy_date).days >= INTERVAL_DAYS:
                    batches += 1
                    should_buy = True

                if should_buy:
                    bp = p * (1 + SLIPPAGE) * (1 + COMMISSION)
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


def grid_search_interval(train_df):
    """Grid search using interval buy training."""
    best_ret = -float('inf')
    best = None
    for a in AND_SELL_RANGE:
        for o in OR_SELL_RANGE:
            fv = run_train_interval(train_df, a, o)
            ret = (fv / INITIAL_CAPITAL - 1) * 100
            if ret > best_ret:
                best_ret = ret
                best = (a, o)
    return best, best_ret


# ============================================================
# Test: Interval buy (same for both modes)
# ============================================================

def run_interval(df, and_t, or_t, cash, position, entry_price,
                 in_buy_mode=False, last_buy_date=None, buy_base=None, batches=0):
    """Test-period interval buy execution with state carryover."""
    trades = []

    for i in range(len(df)):
        dt = df.index[i]
        p = df['Close'].iloc[i]
        s = df['sentiment'].iloc[i]
        ma = df['MA50'].iloc[i]

        if position > 0:
            sell = False
            reason = ""
            if s > or_t:
                sell = True
                reason = f"OR: {s:.0f}>{or_t}"
            elif s > and_t and p < ma:
                sell = True
                reason = f"AND: {s:.0f}>{and_t}&P<MA50"
            if sell:
                sp = p * (1 - SLIPPAGE) * (1 - COMMISSION)
                pnl = (sp - entry_price) / entry_price * 100 if entry_price > 0 else 0
                cash += position * sp
                trades.append({'type': 'SELL', 'date': dt, 'price': p, 'shares': position,
                               'profit_pct': pnl, 'reason': reason})
                position = 0
                entry_price = 0
                in_buy_mode = False
                last_buy_date = None
                buy_base = None
                batches = 0

        if position == 0 or in_buy_mode:
            if not in_buy_mode and s < INTERVAL_BUY_THRESHOLD:
                in_buy_mode = True
                buy_base = cash + position * p
                batches = 0
                last_buy_date = None

            if in_buy_mode:
                should_buy = False
                buy_all = False
                reason = ""
                if last_buy_date is None:
                    should_buy = True
                    reason = f"B1: {s:.0f}<{INTERVAL_BUY_THRESHOLD}"
                elif s >= INTERVAL_BUY_THRESHOLD:
                    should_buy = True
                    buy_all = True
                    reason = f"ALL: {s:.0f}>={INTERVAL_BUY_THRESHOLD}"
                elif (dt - last_buy_date).days >= INTERVAL_DAYS:
                    batches += 1
                    should_buy = True
                    reason = f"B{batches+1}: +{(dt - last_buy_date).days}d"

                if should_buy:
                    bp = p * (1 + SLIPPAGE) * (1 + COMMISSION)
                    shares = int(cash / bp) if buy_all else int(buy_base * INTERVAL_BATCH_PCT / bp)
                    if shares > 0 and cash >= shares * bp:
                        cost = shares * bp
                        entry_price = (entry_price * position + cost) / (position + shares) if position > 0 else bp
                        position += shares
                        cash -= cost
                        last_buy_date = dt
                        trades.append({'type': 'BUY', 'date': dt, 'price': p, 'shares': shares, 'reason': reason})
                    if buy_all:
                        in_buy_mode = False
                        buy_base = None
                        batches = 0

    fv = cash + position * df['Close'].iloc[-1] if len(df) > 0 else cash
    return fv, trades, cash, position, entry_price, in_buy_mode, last_buy_date, buy_base, batches


# ============================================================
# Walk-Forward
# ============================================================

def wf_run(symbol, table, use_interval_train):
    """Walk-forward with configurable training mode.
    use_interval_train=False: staged buy training (current)
    use_interval_train=True:  interval buy training (proposed)
    """
    price_df = load_price(symbol)
    sentiment_df = load_sentiment(table, symbol)
    cash = INITIAL_CAPITAL
    pos = 0
    ep = 0
    ibm = False
    lbd = None
    bb = None
    bat = 0
    all_trades = []
    params = []

    for w in WINDOWS:
        train_df = prepare_data(price_df, sentiment_df, w['train'][0], w['train'][1])
        test_df = prepare_data(price_df, sentiment_df, w['test'][0], w['test'][1])
        if len(train_df) < 100 or len(test_df) < 10:
            continue

        # Grid search with selected training mode
        if use_interval_train:
            best, train_ret = grid_search_interval(train_df)
        else:
            best, train_ret = grid_search_staged(train_df)

        start_val = cash + pos * test_df['Close'].iloc[0] if len(test_df) > 0 else cash

        # Test always uses interval buy
        fv, trades, cash, pos, ep, ibm, lbd, bb, bat = run_interval(
            test_df, best[0], best[1], cash, pos, ep, ibm, lbd, bb, bat)

        ret = (fv / start_val - 1) * 100 if start_val > 0 else 0
        params.append({
            'window': w['name'],
            'and': best[0], 'or': best[1],
            'train_ret': train_ret,
            'return': ret, 'end_value': fv
        })
        all_trades.extend(trades)

    total_ret = (fv / INITIAL_CAPITAL - 1) * 100
    return all_trades, params, total_ret, fv


# ============================================================
# Comparison
# ============================================================

def compare_stock(symbol, table_name, display):
    print(f"\n{'='*65}")
    print(f"{symbol} [{display}]  Train Mode Comparison (Interval Buy Test)")
    print(f"{'='*65}")

    # Mode A: staged train
    trades_a, params_a, ret_a, fv_a = wf_run(symbol, table_name, use_interval_train=False)
    # Mode B: interval train
    trades_b, params_b, ret_b, fv_b = wf_run(symbol, table_name, use_interval_train=True)

    # Window detail
    print(f"\n{'Window':<8} {'--- Staged Train ---':>24} {'--- Interval Train ---':>26}")
    print(f"{'':>8} {'AND':>5} {'OR':>5} {'Test%':>8} {'AND':>7} {'OR':>5} {'Test%':>8} {'Diff':>8}")
    print("-" * 62)
    for pa, pb in zip(params_a, params_b):
        diff = pb['return'] - pa['return']
        same = "=" if pa['and'] == pb['and'] and pa['or'] == pb['or'] else "*"
        print(f"{pa['window']:<8} {'>'+str(pa['and']):>5} {'>'+str(pa['or']):>5} {pa['return']:>+7.1f}%"
              f"  {same} {'>'+str(pb['and']):>5} {'>'+str(pb['or']):>5} {pb['return']:>+7.1f}% {diff:>+7.1f}%")

    diff_total = ret_b - ret_a
    diff_val = fv_b - fv_a
    winner = "Interval" if diff_total > 0 else ("Staged" if diff_total < 0 else "Tie")
    print(f"\n{'Total':>8} {ret_a:>+15.1f}%   {ret_b:>+18.1f}%  {diff_total:>+7.1f}%")
    print(f"{'Value':>8} ${fv_a:>14,.0f}   ${fv_b:>17,.0f}  {'+' if diff_val >= 0 else ''}${diff_val:,.0f}")
    print(f"  Winner: {winner} train")

    # Count parameter differences
    changed = sum(1 for a, b in zip(params_a, params_b) if a['and'] != b['and'] or a['or'] != b['or'])
    print(f"  Params changed: {changed}/{len(params_a)} windows")

    return {
        'symbol': symbol,
        'staged_ret': ret_a, 'staged_val': fv_a, 'staged_params': params_a,
        'interval_ret': ret_b, 'interval_val': fv_b, 'interval_params': params_b,
        'diff_ret': diff_total, 'diff_val': diff_val, 'winner': winner,
        'changed': changed,
    }


def plot_comparison(results, display):
    """Bar chart comparing staged vs interval training for all stocks."""
    symbols = [r['symbol'] for r in results]
    staged = [r['staged_ret'] for r in results]
    interval = [r['interval_ret'] for r in results]

    x = np.arange(len(symbols))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))
    bars1 = ax.bar(x - width/2, staged, width, label='Staged Train (current)',
                   color='#4477aa', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, interval, width, label='Interval Train (proposed)',
                   color='#cc6633', alpha=0.8, edgecolor='black', linewidth=0.5)

    # Value labels
    for bar, val in zip(bars1, staged):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val:+.0f}%', ha='center', va='bottom', fontsize=9, color='#4477aa', fontweight='bold')
    for bar, val in zip(bars2, interval):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val:+.0f}%', ha='center', va='bottom', fontsize=9, color='#cc6633', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(symbols, fontsize=12)
    ax.set_ylabel('Total Return (%) [2020-2025]', fontsize=12)
    ax.set_title(f'Sell Param Training Mode: Staged vs Interval Buy [{display}]\n'
                 f'(Both use interval buy for testing)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)

    # Summary text
    total_staged = sum(r['staged_val'] for r in results)
    total_interval = sum(r['interval_val'] for r in results)
    diff = total_interval - total_staged
    wins_i = sum(1 for r in results if r['diff_ret'] > 0)
    wins_s = len(results) - wins_i
    ax.text(0.02, 0.97,
            f"Staged total: ${total_staged:,.0f}\n"
            f"Interval total: ${total_interval:,.0f}\n"
            f"Diff: {'+' if diff >= 0 else ''}${diff:,.0f}\n"
            f"Score: Staged {wins_s} : {wins_i} Interval",
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    out = f'train_mode_comparison_{display.lower()}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out}")


def plot_param_diff(results, display):
    """Show which windows got different AND/OR params from interval training."""
    fig, ax = plt.subplots(figsize=(16, 6))

    symbols = [r['symbol'] for r in results]
    window_names = [w['name'] for w in WINDOWS]

    for i, r in enumerate(results):
        for j, (pa, pb) in enumerate(zip(r['staged_params'], r['interval_params'])):
            same = pa['and'] == pb['and'] and pa['or'] == pb['or']
            if same:
                ax.scatter(j, i, color='#aaaaaa', marker='s', s=200, zorder=3)
                ax.text(j, i, f"A>{pa['and']}\nO>{pa['or']}", ha='center', va='center',
                        fontsize=7, color='#666666')
            else:
                diff = pb['return'] - pa['return']
                color = '#2ca02c' if diff > 0 else '#d62728'
                ax.scatter(j, i, color=color, marker='s', s=200, zorder=3, alpha=0.8)
                ax.text(j, i, f"{pa['and']}/{pa['or']}\n->{pb['and']}/{pb['or']}\n{diff:+.1f}%",
                        ha='center', va='center', fontsize=6.5, color=color, fontweight='bold')

    ax.set_xticks(range(len(window_names)))
    ax.set_xticklabels(window_names, fontsize=11)
    ax.set_yticks(range(len(symbols)))
    ax.set_yticklabels(symbols, fontsize=11)
    ax.set_title(f'Parameter Changes: Staged -> Interval Training [{display}]\n'
                 f'(Gray=same, Green=improved, Red=worse)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-0.5, len(window_names) - 0.5)
    ax.set_ylim(-0.5, len(symbols) - 0.5)

    plt.tight_layout()
    out = f'param_diff_{display.lower()}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_mode_comparison.py SYMBOL [TABLE]")
        print("  SYMBOL: NVDA, TSLA, ..., ALL")
        print("  TABLE: s3 (default), s5, mf26")
        sys.exit(1)

    symbol = sys.argv[1].upper()
    table_key = sys.argv[2] if len(sys.argv) > 2 else "s3"
    table_name, display = resolve_table(table_key)

    if symbol == "ALL":
        symbols = SYMBOLS
    else:
        symbols = [symbol]

    results = []
    for s in symbols:
        r = compare_stock(s, table_name, display)
        results.append(r)

    # Summary
    if len(results) > 1:
        print(f"\n\n{'='*75}")
        print(f"SUMMARY: Staged vs Interval Training [{display}]")
        print(f"{'='*75}")
        print(f"{'Stock':<8} {'Staged Train':>14} {'Interval Train':>16} {'Diff':>10} {'Params':>8} {'Winner':>10}")
        print(f"{'-'*68}")

        total_s = total_i = 0
        wins_s = wins_i = 0
        for r in results:
            w = "Interval" if r['diff_ret'] > 0 else ("Staged" if r['diff_ret'] < 0 else "Tie")
            if r['diff_ret'] > 0:
                wins_i += 1
            elif r['diff_ret'] < 0:
                wins_s += 1
            print(f"{r['symbol']:<8} {r['staged_ret']:>+13.1f}% {r['interval_ret']:>+15.1f}% "
                  f"{r['diff_ret']:>+9.1f}% {r['changed']:>5}/6 {w:>10}")
            total_s += r['staged_val']
            total_i += r['interval_val']

        diff = total_i - total_s
        diff_pct = (total_i / total_s - 1) * 100 if total_s > 0 else 0
        print(f"{'-'*68}")
        print(f"{'Total $':<8} ${total_s:>12,.0f} ${total_i:>14,.0f} "
              f"{'+' if diff >= 0 else ''}${diff:>8,.0f}")
        print(f"{'Score':<8} {'':>14} {'':>16} {'':>10} {'':>8} {wins_s}:{wins_i}")
        print(f"\nInterval train vs Staged train: {'+' if diff_pct >= 0 else ''}{diff_pct:.1f}% "
              f"({'+' if diff >= 0 else ''}${diff:,.0f})")

        # Charts
        plot_comparison(results, display)
        plot_param_diff(results, display)
    else:
        r = results[0]
        print(f"\nResult: Staged {r['staged_ret']:+.1f}% vs Interval {r['interval_ret']:+.1f}% "
              f"-> Diff {r['diff_ret']:+.1f}%")
