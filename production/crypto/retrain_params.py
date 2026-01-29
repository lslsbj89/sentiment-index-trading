"""
Crypto Sell Parameter Quarterly Retraining
==========================================
Trains optimal AND/OR sell thresholds for BTC, ETH, SOL.
Uses interval buy training (matches production strategy).

Usage:
  python3 retrain_sell_params.py              # Auto: latest data
  python3 retrain_sell_params.py 2023 2025    # Manual: specify train years

Output:
  sell_params_YYYYMMDD.csv  (per-coin optimal sell parameters)

Schedule: Run every 3 months (Jan, Apr, Jul, Oct)
"""

import sys
import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime

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

# Per-coin buy parameters (fixed, from threshold search)
COIN_CONFIG = {
    'BTC': {'buy_threshold': 10,  'train_years': 3},
    'ETH': {'buy_threshold': -10, 'train_years': 3},
    'SOL': {'buy_threshold': -10, 'train_years': 2},
}

# Interval buy parameters
INTERVAL_DAYS = 7
INTERVAL_BATCH_PCT = 0.20

# Sell parameter search grid (crypto: extended OR range)
AND_SELL_RANGE = [5, 10, 15, 20, 25]
OR_SELL_RANGE = [30, 50, 70, 90, 120, 150, 200]

SYMBOLS = ["BTC", "ETH", "SOL"]


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
# Interval Buy Training (matches production buy strategy)
# ============================================================

def run_train_interval(df, and_t, or_t, buy_threshold):
    """Simulate interval buy strategy on training data. Returns final value."""
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

        # Sell check
        if position > 0:
            if s > or_t or (s > and_t and p < ma):
                cash += position * p * (1 - SLIPPAGE) * (1 - COMMISSION)
                position = 0
                entry_price = 0
                in_buy_mode = False
                last_buy_date = None
                buy_base = None
                batches = 0

        # Buy check
        if position == 0 or in_buy_mode:
            if not in_buy_mode and s < buy_threshold:
                in_buy_mode = True
                buy_base = cash + position * p
                batches = 0
                last_buy_date = None

            if in_buy_mode:
                should_buy = False
                buy_all = False
                if last_buy_date is None:
                    should_buy = True
                elif s >= buy_threshold:
                    should_buy = True
                    buy_all = True
                elif (dt - last_buy_date).days >= INTERVAL_DAYS:
                    batches += 1
                    should_buy = True

                if should_buy:
                    bp = p * (1 + SLIPPAGE) * (1 + COMMISSION)
                    if buy_all:
                        shares = cash / bp  # 加密货币支持小数
                    else:
                        shares = buy_base * INTERVAL_BATCH_PCT / bp  # 加密货币支持小数
                    if shares > 0.0001 and cash >= shares * bp:  # 最小交易量 0.0001
                        cost = shares * bp
                        if position > 0:
                            entry_price = (entry_price * position + cost) / (position + shares)
                        else:
                            entry_price = bp
                        position += shares
                        cash -= cost
                        last_buy_date = dt
                    if buy_all:
                        in_buy_mode = False
                        buy_base = None
                        batches = 0

    return cash + position * df['Close'].iloc[-1] if len(df) > 0 else cash


def grid_search(train_df, buy_threshold):
    """Search AND x OR grid (35 combinations). Returns best params and return."""
    best_return = -float('inf')
    best_params = None
    results = []

    for and_t in AND_SELL_RANGE:
        for or_t in OR_SELL_RANGE:
            fv = run_train_interval(train_df, and_t, or_t, buy_threshold)
            ret = (fv / INITIAL_CAPITAL - 1) * 100
            results.append((and_t, or_t, ret))
            if ret > best_return:
                best_return = ret
                best_params = (and_t, or_t)

    return best_params, best_return, results


# ============================================================
# Current Signal Check
# ============================================================

def check_current_signal(price_df, sentiment_df, symbol, and_t, or_t):
    """Check today's signal status for a coin."""
    config = COIN_CONFIG[symbol]
    buy_threshold = config['buy_threshold']

    # Get latest data
    df = price_df.copy()
    df['sentiment'] = sentiment_df['smoothed_index'].reindex(df.index)
    df['MA50'] = df['Close'].rolling(50).mean()
    df = df.dropna()

    if len(df) == 0:
        return None

    latest = df.iloc[-1]
    date = df.index[-1]
    price = latest['Close']
    sent = latest['sentiment']
    ma50 = latest['MA50']

    # Determine signal
    sell_or = sent > or_t
    sell_and = sent > and_t and price < ma50
    buy_trigger = sent < buy_threshold

    if sell_or:
        signal = "SELL (OR)"
        reason = f"sent {sent:.1f} > {or_t}"
    elif sell_and:
        signal = "SELL (AND)"
        reason = f"sent {sent:.1f} > {and_t} & price < MA50"
    elif buy_trigger:
        signal = "BUY MODE"
        reason = f"sent {sent:.1f} < {buy_threshold}"
    else:
        signal = "HOLD"
        reason = f"sent {sent:.1f}, no trigger"

    return {
        'date': date.strftime('%Y-%m-%d'),
        'price': price,
        'sentiment': sent,
        'ma50': ma50,
        'signal': signal,
        'reason': reason,
    }


# ============================================================
# Main
# ============================================================

def main():
    # Parse command-line arguments
    if len(sys.argv) >= 3:
        train_start_year = int(sys.argv[1])
        train_end_year = int(sys.argv[2])
        manual_years = True
    else:
        now = datetime.now()
        train_end_year = now.year - 1
        manual_years = False

    timestamp = datetime.now().strftime('%Y%m%d')

    print("=" * 65)
    print("Crypto Sell Parameter Training (Interval Buy)")
    print("=" * 65)
    print(f"AND search: {AND_SELL_RANGE}")
    print(f"OR search:  {OR_SELL_RANGE}")
    print(f"Grid: {len(AND_SELL_RANGE)} x {len(OR_SELL_RANGE)} = "
          f"{len(AND_SELL_RANGE) * len(OR_SELL_RANGE)} combinations")
    print()

    rows = []
    signals = []

    for symbol in SYMBOLS:
        config = COIN_CONFIG[symbol]
        buy_threshold = config['buy_threshold']
        train_years = config['train_years']

        # Calculate training period
        if manual_years:
            ts = f"{train_start_year}-01-01"
            te = f"{train_end_year}-12-31"
        else:
            ts = f"{train_end_year - train_years + 1}-01-01"
            te = f"{train_end_year}-12-31"

        print(f"--- {symbol} (buy<{buy_threshold}, {train_years}yr train: {ts} ~ {te}) ---")

        # Load data
        price_df = load_crypto_price(symbol)
        sentiment_df = load_crypto_sentiment(symbol)
        train_df = prepare_data(price_df, sentiment_df, ts, te)

        if len(train_df) < 100:
            print(f"  Data insufficient ({len(train_df)} rows), skipping")
            continue

        # Grid search
        best_params, best_return, results = grid_search(train_df, buy_threshold)
        and_t, or_t = best_params

        # Top 3 results
        top3 = sorted(results, key=lambda x: x[2], reverse=True)[:3]

        print(f"  BEST: AND>{and_t}, OR>{or_t} -> +{best_return:.1f}%")
        print(f"  Top3: ", end="")
        for rank, (a, o, r) in enumerate(top3):
            sep = "  |  " if rank > 0 else ""
            print(f"{sep}A>{a}/O>{o} +{r:.1f}%", end="")
        print()

        # Current signal
        sig = check_current_signal(price_df, sentiment_df, symbol, and_t, or_t)
        if sig:
            signals.append((symbol, sig))
            print(f"  Signal: {sig['signal']} ({sig['reason']})")
            print(f"    {sig['date']}: ${sig['price']:,.2f}  sent={sig['sentiment']:.1f}  MA50=${sig['ma50']:,.2f}")
        print()

        rows.append({
            'symbol': symbol,
            'and_threshold': and_t,
            'or_threshold': or_t,
            'buy_threshold': buy_threshold,
            'train_years': train_years,
            'train_return': round(best_return, 1),
            'train_period': f"{ts[:4]}-{te[:4]}",
        })

    # Summary table
    print("=" * 65)
    print("SELL PARAMETERS SUMMARY")
    print("=" * 65)
    print(f"{'Coin':<6} {'Buy<':>6} {'AND':>6} {'OR':>6} {'Train':>10} {'Period':>12}")
    print("-" * 50)
    for r in rows:
        print(f"{r['symbol']:<6} {r['buy_threshold']:>6} "
              f"{'>' + str(r['and_threshold']):>6} "
              f"{'>' + str(r['or_threshold']):>6} "
              f"{r['train_return']:>+9.1f}% "
              f"{r['train_period']:>12}")

    # Current signals
    if signals:
        print()
        print("=" * 65)
        print("CURRENT SIGNALS")
        print("=" * 65)
        print(f"{'Coin':<6} {'Price':>12} {'Sent':>8} {'MA50':>12} {'Signal':<16} {'Reason'}")
        print("-" * 75)
        for symbol, sig in signals:
            print(f"{symbol:<6} ${sig['price']:>10,.2f} {sig['sentiment']:>7.1f} "
                  f"${sig['ma50']:>10,.2f} {sig['signal']:<16} {sig['reason']}")

    # Save CSV
    fname = f"params_{timestamp}.csv"
    pd.DataFrame(rows).to_csv(fname, index=False)
    print(f"\nSaved: {fname}")
    print(f"Next update: rerun in 3 months")


if __name__ == "__main__":
    main()
