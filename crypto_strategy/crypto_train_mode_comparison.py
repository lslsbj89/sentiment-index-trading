"""
Crypto Train Mode Comparison
==============================
Compares 3 training modes for sell parameter grid search:

  1. Baseline (staged):  Training uses staged buy [5,0,-5,-10] (current code)
  2. Mode A (interval):  Training uses interval buy (matches test strategy)
  3. Mode B (oneshot):   Training uses one-shot buy/sell (simplified)

All 3 modes use the SAME interval buy strategy for TESTING.
The only difference is how sell parameters (AND/OR) are optimized during training.

Usage:
  python crypto_train_mode_comparison.py BTC
  python crypto_train_mode_comparison.py ALL
"""

import sys
import pandas as pd
import numpy as np
import psycopg2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

INTERVAL_DAYS = 7
INTERVAL_BATCH_PCT = 0.20

# Buy threshold search space
BUY_THRESHOLD_CANDIDATES = [-10, -5, 0, 5, 10, 15, 20]

# Sell grid
AND_SELL_RANGE = [5, 10, 15, 20, 25]
OR_SELL_RANGE = [30, 50, 70, 90, 120, 150, 200]

# Staged buy params (for baseline training)
STAGED_BUY_THRESHOLDS = [5, 0, -5, -10]
STAGED_BATCH_PCT = 0.25

# Per-coin windows (test: 2023-2025)
COIN_WINDOWS = {
    'BTC': [
        {"name": "W2023", "train": ("2020-01-01", "2022-12-31"), "test": ("2023-01-01", "2023-12-31")},
        {"name": "W2024", "train": ("2021-01-01", "2023-12-31"), "test": ("2024-01-01", "2024-12-31")},
        {"name": "W2025", "train": ("2022-01-01", "2024-12-31"), "test": ("2025-01-01", "2025-12-31")},
    ],
    'ETH': [
        {"name": "W2023", "train": ("2020-01-01", "2022-12-31"), "test": ("2023-01-01", "2023-12-31")},
        {"name": "W2024", "train": ("2021-01-01", "2023-12-31"), "test": ("2024-01-01", "2024-12-31")},
        {"name": "W2025", "train": ("2022-01-01", "2024-12-31"), "test": ("2025-01-01", "2025-12-31")},
    ],
    'SOL': [
        {"name": "W2023", "train": ("2021-01-01", "2022-12-31"), "test": ("2023-01-01", "2023-12-31")},
        {"name": "W2024", "train": ("2022-01-01", "2023-12-31"), "test": ("2024-01-01", "2024-12-31")},
        {"name": "W2025", "train": ("2023-01-01", "2024-12-31"), "test": ("2025-01-01", "2025-12-31")},
    ],
}


# ============================================================
# Data Loading
# ============================================================

def load_crypto_price(symbol):
    conn = psycopg2.connect(**DB_CONFIG)
    query = """
        SELECT open_time, open_price::float, high_price::float,
               low_price::float, close_price::float, volume::float
        FROM yahoo_candles
        WHERE symbol = %s AND interval = '1d'
        ORDER BY open_time
    """
    df = pd.read_sql(query, conn, params=(symbol,))
    df['date'] = pd.to_datetime(df['open_time'] / 1000, unit='s', utc=True)
    df = df.set_index('date')
    df = df.rename(columns={
        'open_price': 'Open', 'high_price': 'High',
        'low_price': 'Low', 'close_price': 'Close',
        'volume': 'Volume'
    })
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    conn.close()
    return df


def load_crypto_sentiment(symbol):
    conn = psycopg2.connect(**DB_CONFIG)
    query = """
        SELECT date, smoothed_index::float
        FROM yahoo_artemis_index
        WHERE symbol = %s
        ORDER BY date
    """
    df = pd.read_sql(query, conn, params=(symbol,))
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.set_index('date')
    conn.close()
    return df


def prepare_data(price_df, sentiment_df, start_date, end_date):
    df = price_df.copy()
    df['sentiment'] = sentiment_df['smoothed_index'].reindex(df.index)
    df['MA50'] = df['Close'].rolling(50).mean()
    df = df.dropna()
    return df[start_date:end_date]


# ============================================================
# Training Mode 1: Staged Buy (Baseline, current code)
# ============================================================

def run_train_staged(df, and_t, or_t):
    cash = INITIAL_CAPITAL
    position = 0; entry_price = 0
    bought_levels = set(); base = cash

    for i in range(len(df)):
        p = df['Close'].iloc[i]; s = df['sentiment'].iloc[i]; ma = df['MA50'].iloc[i]

        if position > 0:
            if s > or_t or (s > and_t and p < ma):
                cash += position * p * (1 - SLIPPAGE) * (1 - COMMISSION)
                position = 0; entry_price = 0; bought_levels = set(); base = cash

        for idx, th in enumerate(STAGED_BUY_THRESHOLDS):
            if idx not in bought_levels and s < th:
                bp = p * (1 + SLIPPAGE) * (1 + COMMISSION)
                shares = base * STAGED_BATCH_PCT / bp  # 加密货币支持小数
                if shares > 0.0001 and cash >= shares * bp:  # 最小交易量 0.0001
                    cost = shares * bp
                    entry_price = (entry_price * position + cost) / (position + shares) if position > 0 else bp
                    position += shares; cash -= cost; bought_levels.add(idx)

    return cash + position * df['Close'].iloc[-1] if len(df) > 0 else cash


# ============================================================
# Training Mode 2: Interval Buy (matches test strategy)
# ============================================================

def run_train_interval(df, and_t, or_t, buy_threshold):
    cash = INITIAL_CAPITAL
    position = 0; entry_price = 0
    in_buy_mode = False; last_buy_date = None; buy_base = None; batches = 0

    for i in range(len(df)):
        dt = df.index[i]; p = df['Close'].iloc[i]; s = df['sentiment'].iloc[i]; ma = df['MA50'].iloc[i]

        if position > 0:
            if s > or_t or (s > and_t and p < ma):
                cash += position * p * (1 - SLIPPAGE) * (1 - COMMISSION)
                position = 0; entry_price = 0
                in_buy_mode = False; last_buy_date = None; buy_base = None; batches = 0

        if position == 0 or in_buy_mode:
            if not in_buy_mode and s < buy_threshold:
                in_buy_mode = True; buy_base = cash + position * p; batches = 0; last_buy_date = None

            if in_buy_mode:
                should_buy = False; buy_all = False
                if last_buy_date is None:
                    should_buy = True
                elif s >= buy_threshold:
                    should_buy = True; buy_all = True
                elif (dt - last_buy_date).days >= INTERVAL_DAYS:
                    batches += 1; should_buy = True

                if should_buy:
                    bp = p * (1 + SLIPPAGE) * (1 + COMMISSION)
                    shares = cash / bp if buy_all else buy_base * INTERVAL_BATCH_PCT / bp  # 加密货币支持小数
                    if shares > 0.0001 and cash >= shares * bp:  # 最小交易量 0.0001
                        cost = shares * bp
                        entry_price = (entry_price * position + cost) / (position + shares) if position > 0 else bp
                        position += shares; cash -= cost; last_buy_date = dt
                    if buy_all:
                        in_buy_mode = False; buy_base = None; batches = 0

    return cash + position * df['Close'].iloc[-1] if len(df) > 0 else cash


# ============================================================
# Training Mode 3: One-shot Buy/Sell (simplified)
# ============================================================

def run_train_oneshot(df, and_t, or_t, buy_threshold):
    cash = INITIAL_CAPITAL
    position = 0

    for i in range(len(df)):
        p = df['Close'].iloc[i]; s = df['sentiment'].iloc[i]; ma = df['MA50'].iloc[i]

        if position > 0:
            if s > or_t or (s > and_t and p < ma):
                cash += position * p * (1 - SLIPPAGE) * (1 - COMMISSION)
                position = 0

        if position == 0 and s < buy_threshold:
            bp = p * (1 + SLIPPAGE) * (1 + COMMISSION)
            shares = cash / bp  # 加密货币支持小数
            if shares > 0.0001:  # 最小交易量 0.0001
                position = shares; cash -= shares * bp

    return cash + position * df['Close'].iloc[-1] if len(df) > 0 else cash


# ============================================================
# Grid Search (parameterized by training mode)
# ============================================================

def grid_search(train_df, mode, buy_threshold=0):
    """
    mode: 'staged', 'interval', 'oneshot'
    buy_threshold: used for interval and oneshot modes
    """
    best_ret = -float('inf')
    best = None
    for a in AND_SELL_RANGE:
        for o in OR_SELL_RANGE:
            if mode == 'staged':
                fv = run_train_staged(train_df, a, o)
            elif mode == 'interval':
                fv = run_train_interval(train_df, a, o, buy_threshold)
            elif mode == 'oneshot':
                fv = run_train_oneshot(train_df, a, o, buy_threshold)
            ret = (fv / INITIAL_CAPITAL - 1) * 100
            if ret > best_ret:
                best_ret = ret; best = (a, o)
    return best, best_ret


# ============================================================
# Test: Interval Buy (same for all modes)
# ============================================================

def run_test_interval(df, and_t, or_t, buy_threshold, cash, position, entry_price,
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
            if not in_buy_mode and s < buy_threshold:
                in_buy_mode = True; buy_base = cash + position * p; batches = 0; last_buy_date = None

            if in_buy_mode:
                should_buy = False; buy_all = False; reason = ""
                if last_buy_date is None:
                    should_buy = True; reason = f"B1: {s:.0f}<{buy_threshold}"
                elif s >= buy_threshold:
                    should_buy = True; buy_all = True; reason = f"ALL: {s:.0f}>={buy_threshold}"
                elif (dt - last_buy_date).days >= INTERVAL_DAYS:
                    batches += 1; should_buy = True; reason = f"B{batches+1}: +{(dt - last_buy_date).days}d"

                if should_buy:
                    bp = p * (1 + SLIPPAGE) * (1 + COMMISSION)
                    shares = cash / bp if buy_all else buy_base * INTERVAL_BATCH_PCT / bp  # 加密货币支持小数
                    if shares > 0.0001 and cash >= shares * bp:  # 最小交易量 0.0001
                        cost = shares * bp
                        entry_price = (entry_price * position + cost) / (position + shares) if position > 0 else bp
                        position += shares; cash -= cost; last_buy_date = dt
                        trades.append({'type': 'BUY', 'date': dt, 'price': p, 'shares': shares, 'reason': reason})
                    if buy_all:
                        in_buy_mode = False; buy_base = None; batches = 0

    fv = cash + position * df['Close'].iloc[-1] if len(df) > 0 else cash
    return fv, trades, cash, position, entry_price, in_buy_mode, last_buy_date, buy_base, batches


# ============================================================
# Walk-Forward with mode selection
# ============================================================

def wf_interval_with_mode(symbol, buy_threshold, mode):
    """
    Walk-forward test using interval buy for testing,
    but with configurable training mode for sell parameter optimization.
    """
    windows = COIN_WINDOWS[symbol]
    price_df = load_crypto_price(symbol)
    sentiment_df = load_crypto_sentiment(symbol)
    cash = INITIAL_CAPITAL; pos = 0; ep = 0
    ibm = False; lbd = None; bb = None; bat = 0
    all_trades = []; params = []

    for w in windows:
        train_df = prepare_data(price_df, sentiment_df, w['train'][0], w['train'][1])
        test_df = prepare_data(price_df, sentiment_df, w['test'][0], w['test'][1])
        if len(train_df) < 50 or len(test_df) < 10:
            continue
        best, train_ret = grid_search(train_df, mode, buy_threshold)
        start_val = cash + pos * test_df['Close'].iloc[0] if len(test_df) > 0 else cash
        fv, trades, cash, pos, ep, ibm, lbd, bb, bat = run_test_interval(
            test_df, best[0], best[1], buy_threshold, cash, pos, ep, ibm, lbd, bb, bat)
        ret = (fv / start_val - 1) * 100 if start_val > 0 else 0
        params.append({'window': w['name'], 'and': best[0], 'or': best[1], 'return': ret, 'end_value': fv})
        all_trades.extend(trades)

    total_ret = (fv / INITIAL_CAPITAL - 1) * 100
    buys = len([t for t in all_trades if t['type'] == 'BUY'])
    sells = len([t for t in all_trades if t['type'] == 'SELL'])
    return total_ret, fv, params, buys, sells


# ============================================================
# Search for best threshold under each training mode
# ============================================================

def search_coin_mode(symbol, mode):
    results = []
    for bt in BUY_THRESHOLD_CANDIDATES:
        total_ret, fv, params, buys, sells = wf_interval_with_mode(symbol, bt, mode)
        results.append({
            'threshold': bt,
            'total_ret': total_ret,
            'final_value': fv,
            'buys': buys,
            'sells': sells,
            'params': params
        })
    best = max(results, key=lambda x: x['total_ret'])
    return results, best


# ============================================================
# Main
# ============================================================

MODE_NAMES = {
    'staged':   'Baseline (staged train)',
    'interval': 'Mode A (interval train)',
    'oneshot':  'Mode B (oneshot train)',
}


def run_coin(symbol):
    print(f"\n{'='*75}")
    print(f"  {symbol} — Train Mode Comparison (Interval Buy Test)")
    print(f"  Thresholds: {BUY_THRESHOLD_CANDIDATES}")
    print(f"  Sell grid: AND {AND_SELL_RANGE} x OR {OR_SELL_RANGE}")
    print(f"{'='*75}")

    all_mode_results = {}

    for mode in ['staged', 'interval', 'oneshot']:
        print(f"\n--- {MODE_NAMES[mode]} ---")
        results, best = search_coin_mode(symbol, mode)
        all_mode_results[mode] = (results, best)

        for r in results:
            wins = len([p for p in r['params'] if p['return'] > 0])
            print(f"  buy<{r['threshold']:>3}: {r['total_ret']:>+8.1f}%  ${r['final_value']:>10,.0f}  "
                  f"buys={r['buys']:>2} sells={r['sells']:>2}  wins={wins}/{len(r['params'])}")
        print(f"  BEST: buy<{best['threshold']} -> +{best['total_ret']:.1f}% (${best['final_value']:,.0f})")

        # Window details for best
        for p in best['params']:
            print(f"    {p['window']}: AND>{p['and']}, OR>{p['or']} -> {p['return']:+.1f}%")

    # ---- Comparison Table ----
    print(f"\n{'='*75}")
    print(f"  {symbol} — COMPARISON SUMMARY")
    print(f"{'='*75}")
    print(f"  {'Mode':<28} {'Best Thresh':>12} {'Return':>10} {'Final Value':>14}")
    print(f"  {'-'*64}")
    for mode in ['staged', 'interval', 'oneshot']:
        _, best = all_mode_results[mode]
        print(f"  {MODE_NAMES[mode]:<28} {'buy<' + str(best['threshold']):>12} "
              f"{best['total_ret']:>+9.1f}% ${best['final_value']:>12,.0f}")

    # ---- Window-level comparison for each mode's best threshold ----
    print(f"\n  Window-level (each mode's best threshold):")
    print(f"  {'Window':<8}", end="")
    for mode in ['staged', 'interval', 'oneshot']:
        _, best = all_mode_results[mode]
        print(f"  {'[' + mode[:3].upper() + ' buy<' + str(best['threshold']) + ']':>20}", end="")
    print()
    print(f"  {'-'*68}")

    num_windows = len(all_mode_results['staged'][1]['params'])
    for wi in range(num_windows):
        wname = all_mode_results['staged'][1]['params'][wi]['window']
        print(f"  {wname:<8}", end="")
        for mode in ['staged', 'interval', 'oneshot']:
            _, best = all_mode_results[mode]
            p = best['params'][wi]
            print(f"  AND>{p['and']:>2} OR>{p['or']:>3} {p['return']:>+7.1f}%", end="")
        print()

    return all_mode_results


def plot_comparison(all_coin_results):
    coins = list(all_coin_results.keys())
    modes = ['staged', 'interval', 'oneshot']
    mode_colors = {'staged': '#2255aa', 'interval': '#aa5522', 'oneshot': '#22aa55'}
    mode_labels = {'staged': 'Staged Train', 'interval': 'Interval Train', 'oneshot': 'Oneshot Train'}

    fig, axes = plt.subplots(1, len(coins), figsize=(7 * len(coins), 6))
    if len(coins) == 1:
        axes = [axes]

    for idx, coin in enumerate(coins):
        ax = axes[idx]
        mode_results = all_coin_results[coin]

        x = np.arange(len(BUY_THRESHOLD_CANDIDATES))
        width = 0.25

        for mi, mode in enumerate(modes):
            results, best = mode_results[mode]
            returns = [r['total_ret'] for r in results]
            bars = ax.bar(x + (mi - 1) * width, returns, width,
                         label=f"{mode_labels[mode]} (best: buy<{best['threshold']})",
                         color=mode_colors[mode], alpha=0.7, edgecolor='black', linewidth=0.5)

            # Highlight best
            best_idx = BUY_THRESHOLD_CANDIDATES.index(best['threshold'])
            bars[best_idx].set_edgecolor('gold')
            bars[best_idx].set_linewidth(2.5)

        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_xlabel('Buy Threshold (sent < X)', fontsize=11)
        ax.set_ylabel('Total Return (%)', fontsize=11)
        ax.set_title(f'{coin}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([str(t) for t in BUY_THRESHOLD_CANDIDATES])
        ax.legend(fontsize=8, loc='best')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Train Mode Comparison: Staged vs Interval vs Oneshot (Test: Interval Buy)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = 'crypto_train_mode_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python crypto_train_mode_comparison.py SYMBOL")
        print("  SYMBOL: BTC, ETH, SOL, or ALL")
        sys.exit(1)

    symbol = sys.argv[1].upper()

    if symbol == "ALL":
        coins = ["BTC", "ETH", "SOL"]
    else:
        coins = [symbol]

    all_coin_results = {}
    for c in coins:
        all_coin_results[c] = run_coin(c)

    plot_comparison(all_coin_results)

    # ---- Grand Summary ----
    if len(coins) > 1:
        print(f"\n\n{'='*75}")
        print(f"  GRAND SUMMARY — All Coins x All Modes")
        print(f"{'='*75}")
        print(f"  {'Coin':<6} {'Staged Train':>20} {'Interval Train':>20} {'Oneshot Train':>20}")
        print(f"  {'-'*66}")
        totals = {'staged': 0, 'interval': 0, 'oneshot': 0}
        for c in coins:
            mode_results = all_coin_results[c]
            parts = []
            for mode in ['staged', 'interval', 'oneshot']:
                _, best = mode_results[mode]
                parts.append(f"buy<{best['threshold']:>3} {best['total_ret']:>+7.1f}%")
                totals[mode] += best['final_value']
            print(f"  {c:<6} {parts[0]:>20} {parts[1]:>20} {parts[2]:>20}")
        print(f"  {'-'*66}")
        print(f"  {'Total $':<6} ${totals['staged']:>18,.0f} ${totals['interval']:>18,.0f} ${totals['oneshot']:>18,.0f}")

        # Find overall winner
        winner = max(totals, key=totals.get)
        print(f"\n  WINNER: {MODE_NAMES[winner]} (${totals[winner]:,.0f})")
