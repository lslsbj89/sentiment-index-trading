#!/usr/bin/env python3
"""
分析回撤来源：找出最大回撤发生的时间段
"""

import sys
sys.path.insert(0, '/Users/sc2025/Desktop/Claude/Quantrade/src')

import pandas as pd
import numpy as np
import psycopg2
from data_loader import DataLoader
import warnings
warnings.filterwarnings('ignore')

db_config = {
    'host': 'localhost',
    'database': 'crypto_fear_greed_2',
    'user': 'sc2025',
    'password': '',
    'port': 5432
}

SYMBOLS = ['NVDA', 'TSLA', 'META']  # 回撤最大的3只

BUY_THRESHOLD = 5
AND_THRESHOLD = 20
OR_THRESHOLD = 40
INITIAL_CAPITAL = 100000
POSITION_PCT = 0.8


def load_fear_greed_index(symbol):
    conn = psycopg2.connect(**db_config)
    query = f"""
        SELECT date, smoothed_index as idx
        FROM fear_greed_index
        WHERE symbol = '{symbol}'
        ORDER BY date
    """
    df = pd.read_sql(query, conn, parse_dates=['date'])
    conn.close()
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df.set_index('date', inplace=True)
    return df


def analyze_drawdown(symbol):
    """分析单只股票的回撤"""
    loader = DataLoader(db_config)
    prices = loader.load_ohlcv(symbol, start_date='2020-01-01', end_date='2025-12-31')
    loader.close()

    index_data = load_fear_greed_index(symbol)

    df = prices.copy()
    df['idx'] = index_data['idx']
    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()
    df = df.dropna()
    df = df[df.index >= '2021-01-01']

    # 模拟交易
    cash = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    entry_date = None
    portfolio_values = []
    trades = []

    for i in range(len(df)):
        current_price = df['Close'].iloc[i]
        current_idx = df['idx'].iloc[i]
        current_ma50 = df['MA50'].iloc[i]
        current_date = df.index[i]

        if position == 0 and current_idx < BUY_THRESHOLD:
            available = cash * POSITION_PCT
            shares = int(available / (current_price * 1.002))
            if shares > 0:
                cash -= shares * current_price * 1.002
                position = shares
                entry_price = current_price
                entry_date = current_date
                trades.append({
                    'type': 'BUY',
                    'date': current_date,
                    'price': current_price,
                    'idx': current_idx,
                    'price_vs_ma200': 'above' if current_price > df['MA200'].iloc[i] else 'below'
                })

        elif position > 0:
            sell = False
            reason = ''
            if current_idx > OR_THRESHOLD:
                sell, reason = True, 'idx>40'
            elif current_idx > AND_THRESHOLD and current_price < current_ma50:
                sell, reason = True, 'idx>20 & <MA50'

            if sell:
                revenue = position * current_price * 0.998
                profit_pct = (current_price - entry_price) / entry_price * 100
                trades.append({
                    'type': 'SELL',
                    'date': current_date,
                    'price': current_price,
                    'idx': current_idx,
                    'profit_pct': profit_pct,
                    'reason': reason
                })
                cash += revenue
                position = 0

        portfolio_values.append(cash + position * current_price)

    # 期末平仓
    if position > 0:
        final_price = df['Close'].iloc[-1]
        profit_pct = (final_price - entry_price) / entry_price * 100
        trades.append({
            'type': 'SELL',
            'date': df.index[-1],
            'price': final_price,
            'idx': df['idx'].iloc[-1],
            'profit_pct': profit_pct,
            'reason': 'end'
        })

    # 找最大回撤时间段
    portfolio_series = pd.Series(portfolio_values, index=df.index)
    rolling_max = portfolio_series.cummax()
    drawdown = (portfolio_series - rolling_max) / rolling_max * 100

    max_dd_date = drawdown.idxmin()
    max_dd_value = drawdown.min()

    # 找回撤开始日期（高点）
    peak_date = rolling_max[:max_dd_date].idxmax()

    return {
        'symbol': symbol,
        'max_drawdown': max_dd_value,
        'peak_date': peak_date,
        'trough_date': max_dd_date,
        'trades': trades,
        'df': df
    }


def main():
    print("=" * 90)
    print("回撤来源分析")
    print("=" * 90)

    for symbol in SYMBOLS:
        result = analyze_drawdown(symbol)

        print(f"\n{'='*40}")
        print(f"{symbol}: 最大回撤 {result['max_drawdown']:.1f}%")
        print(f"{'='*40}")
        print(f"高点日期: {result['peak_date'].strftime('%Y-%m-%d')}")
        print(f"低点日期: {result['trough_date'].strftime('%Y-%m-%d')}")

        # 找出在回撤期间的持仓情况
        df = result['df']
        peak = result['peak_date']
        trough = result['trough_date']

        print(f"\n回撤期间价格变化:")
        peak_price = df.loc[peak, 'Close']
        trough_price = df.loc[trough, 'Close']
        print(f"  高点价格: ${peak_price:.2f}")
        print(f"  低点价格: ${trough_price:.2f}")
        print(f"  价格跌幅: {(trough_price - peak_price) / peak_price * 100:.1f}%")

        print(f"\n交易记录:")
        for t in result['trades']:
            if t['type'] == 'BUY':
                print(f"  {t['date'].strftime('%Y-%m-%d')} BUY  @ ${t['price']:.2f} (idx={t['idx']:.1f}, {t['price_vs_ma200']} MA200)")
            else:
                pnl = t.get('profit_pct', 0)
                print(f"  {t['date'].strftime('%Y-%m-%d')} SELL @ ${t['price']:.2f} (idx={t['idx']:.1f}, {t['reason']}, {pnl:+.1f}%)")

        # 分析回撤期间是否有卖出机会
        dd_period = df[(df.index >= peak) & (df.index <= trough)]
        print(f"\n回撤期间情绪指数范围:")
        print(f"  最低: {dd_period['idx'].min():.1f}")
        print(f"  最高: {dd_period['idx'].max():.1f}")
        print(f"  有卖出信号(idx>20 & <MA50)的天数: {((dd_period['idx'] > 20) & (dd_period['Close'] < dd_period['MA50'])).sum()}")
        print(f"  有卖出信号(idx>40)的天数: {(dd_period['idx'] > 40).sum()}")


if __name__ == '__main__':
    main()
