"""
Crypto Buy Threshold Search
============================
Search for optimal interval buy threshold for each crypto coin.
Also tests extended OR sell range for high-volatility assets.

Tests: INTERVAL_BUY_THRESHOLD = [-10, -5, 0, 5, 10, 15, 20]
       OR_SELL_RANGE extended to [30, 50, 70, 90, 120, 150, 200]

Usage:
  python crypto_threshold_search.py BTC
  python crypto_threshold_search.py ALL
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

# Search space
BUY_THRESHOLD_CANDIDATES = [-10, -5, 0, 5, 10, 15, 20]

# Extended sell grid (wider OR for crypto)
AND_SELL_RANGE = [5, 10, 15, 20, 25]
OR_SELL_RANGE = [30, 50, 70, 90, 120, 150, 200]

# Staged buy thresholds (for grid search training)
STAGED_BUY_THRESHOLDS = [5, 0, -5, -10]
STAGED_BATCH_PCT = 0.25

# Per-coin windows (test: 2023-2025 only, same as crypto_backtest.py)
COIN_WINDOWS = {
    'BTC': [  # 3-year train
        {"name": "W2023", "train": ("2020-01-01", "2022-12-31"), "test": ("2023-01-01", "2023-12-31")},
        {"name": "W2024", "train": ("2021-01-01", "2023-12-31"), "test": ("2024-01-01", "2024-12-31")},
        {"name": "W2025", "train": ("2022-01-01", "2024-12-31"), "test": ("2025-01-01", "2025-12-31")},
    ],
    'ETH': [  # 3-year train
        {"name": "W2023", "train": ("2020-01-01", "2022-12-31"), "test": ("2023-01-01", "2023-12-31")},
        {"name": "W2024", "train": ("2021-01-01", "2023-12-31"), "test": ("2024-01-01", "2024-12-31")},
        {"name": "W2025", "train": ("2022-01-01", "2024-12-31"), "test": ("2025-01-01", "2025-12-31")},
    ],
    'SOL': [  # 2-year train
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
# Grid Search (sell params, staged buy for training)
# ============================================================

def run_train_grid(df, and_t, or_t):
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
                shares = int(base * STAGED_BATCH_PCT / bp)
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
# Interval Buy with configurable threshold
# ============================================================

def run_interval(df, and_t, or_t, buy_threshold, cash, position, entry_price,
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
# Walk-Forward with threshold
# ============================================================

def wf_interval_threshold(symbol, buy_threshold):
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
        best, train_ret = grid_search(train_df)
        start_val = cash + pos * test_df['Close'].iloc[0] if len(test_df) > 0 else cash
        fv, trades, cash, pos, ep, ibm, lbd, bb, bat = run_interval(
            test_df, best[0], best[1], buy_threshold, cash, pos, ep, ibm, lbd, bb, bat)
        ret = (fv / start_val - 1) * 100 if start_val > 0 else 0
        params.append({'window': w['name'], 'and': best[0], 'or': best[1], 'return': ret, 'end_value': fv})
        all_trades.extend(trades)

    total_ret = (fv / INITIAL_CAPITAL - 1) * 100
    buys = len([t for t in all_trades if t['type'] == 'BUY'])
    sells = len([t for t in all_trades if t['type'] == 'SELL'])
    return total_ret, fv, params, buys, sells


# ============================================================
# Main
# ============================================================

def search_coin(symbol):
    windows = COIN_WINDOWS[symbol]
    print(f"\n{'='*70}")
    print(f"Buy Threshold Search: {symbol} (Interval Buy)")
    print(f"Windows: {windows[0]['name']}-{windows[-1]['name']} ({len(windows)} windows)")
    print(f"Thresholds: {BUY_THRESHOLD_CANDIDATES}")
    print(f"OR range: {OR_SELL_RANGE}")
    print(f"{'='*70}")

    results = []
    for bt in BUY_THRESHOLD_CANDIDATES:
        total_ret, fv, params, buys, sells = wf_interval_threshold(symbol, bt)
        results.append({
            'threshold': bt,
            'total_ret': total_ret,
            'final_value': fv,
            'buys': buys,
            'sells': sells,
            'params': params
        })
        wins = [p for p in params if p['return'] > 0]
        print(f"  buy<{bt:>3}: {total_ret:>+8.1f}%  ${fv:>10,.0f}  buys={buys:>2} sells={sells:>2}  wins={len(wins)}/{len(params)}")

    # Find best
    best = max(results, key=lambda x: x['total_ret'])
    print(f"\n  BEST: buy<{best['threshold']} -> +{best['total_ret']:.1f}% (${best['final_value']:,.0f})")

    # Window details for best
    print(f"\n  Window details (buy<{best['threshold']}):")
    for p in best['params']:
        print(f"    {p['window']}: AND>{p['and']}, OR>{p['or']} -> {p['return']:+.1f}%")

    return results, best


def plot_search_results(all_results):
    fig, axes = plt.subplots(1, len(all_results), figsize=(6 * len(all_results), 5))
    if len(all_results) == 1:
        axes = [axes]

    for idx, (symbol, results, best) in enumerate(all_results):
        ax = axes[idx]
        thresholds = [r['threshold'] for r in results]
        returns = [r['total_ret'] for r in results]

        colors = ['#2ca02c' if r > 0 else '#d62728' for r in returns]
        bars = ax.bar(thresholds, returns, width=3, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

        # Highlight best
        best_idx = thresholds.index(best['threshold'])
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)

        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_xlabel('Buy Threshold (sent < X)', fontsize=11)
        ax.set_ylabel('Total Return (%)', fontsize=11)
        ax.set_title(f'{symbol}\nBest: buy<{best["threshold"]} ({best["total_ret"]:+.1f}%)',
                     fontsize=13, fontweight='bold')
        ax.set_xticks(thresholds)
        ax.grid(axis='y', alpha=0.3)

        for i, (t, r) in enumerate(zip(thresholds, returns)):
            ax.text(t, r + (2 if r >= 0 else -8), f'{r:+.0f}%', ha='center', fontsize=9, fontweight='bold')

    plt.suptitle('Interval Buy Threshold Search (Crypto)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = 'crypto_threshold_search.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python crypto_threshold_search.py SYMBOL")
        print("  SYMBOL: BTC, ETH, SOL, or ALL")
        sys.exit(1)

    symbol = sys.argv[1].upper()

    if symbol == "ALL":
        coins = ["BTC", "ETH", "SOL"]
    else:
        coins = [symbol]

    all_results = []
    for c in coins:
        results, best = search_coin(c)
        all_results.append((c, results, best))

    plot_search_results(all_results)

    if len(coins) > 1:
        print(f"\n\n{'='*50}")
        print(f"OPTIMAL THRESHOLDS SUMMARY")
        print(f"{'='*50}")
        print(f"{'Coin':<6} {'Best Threshold':>16} {'Total Return':>14} {'Final Value':>14}")
        print(f"{'-'*50}")
        for c, results, best in all_results:
            print(f"{c:<6} {'buy<' + str(best['threshold']):>16} {best['total_ret']:>+13.1f}% ${best['final_value']:>12,.0f}")
