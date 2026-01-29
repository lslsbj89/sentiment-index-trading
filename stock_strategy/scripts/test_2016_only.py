#!/usr/bin/env python3
"""
测试2016年单独表现（完全未见数据）
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import pandas as pd
import numpy as np
import psycopg2
from data_loader import DataLoader

db_config = {
    'host': 'localhost',
    'database': 'crypto_fear_greed_2',
    'user': 'sc2025',
    'password': '',
    'port': 5432
}

def load_sentiment_index(symbol, smoothing=3):
    """加载情绪指数"""
    conn = psycopg2.connect(**db_config)
    table_name = "fear_greed_index_s3" if smoothing == 3 else "fear_greed_index"
    query = f"""
        SELECT date, smoothed_index
        FROM {table_name}
        WHERE symbol = '{symbol}'
          AND date >= '2012-01-01'
          AND date <= '2016-12-31'
        ORDER BY date
    """
    df = pd.read_sql(query, conn, parse_dates=['date'])
    conn.close()
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df.set_index('date', inplace=True)
    return df

def backtest_strategy(prices, sentiment, buy_threshold, sell_and_threshold, sell_or_threshold):
    """回测策略"""
    df = prices.copy()
    df['idx'] = sentiment['smoothed_index']
    df['MA50'] = df['Close'].rolling(50).mean()
    df = df.dropna()

    if len(df) < 100:
        return None

    cash = 100000
    position = 0
    portfolio_values = []
    trades = []

    entry_price = 0
    entry_date = None

    for i in range(len(df)):
        current_date = df.index[i]
        current_price = df['Close'].iloc[i]
        current_idx = df['idx'].iloc[i]
        current_ma50 = df['MA50'].iloc[i]

        # 买入
        if position == 0 and current_idx < buy_threshold:
            available = cash * 0.8
            shares = int(available / (current_price * 1.002))
            if shares > 0:
                cost = shares * current_price * 1.002
                cash -= cost
                position = shares
                entry_price = current_price * 1.002
                entry_date = current_date

        # 卖出
        elif position > 0:
            sell_signal = False
            exit_reason = None

            if current_idx > sell_or_threshold:
                sell_signal = True
                exit_reason = 'OR'
            elif current_idx > sell_and_threshold and current_price < current_ma50:
                sell_signal = True
                exit_reason = 'AND'

            if sell_signal:
                revenue = position * current_price * 0.998
                cash += revenue
                profit = revenue - (position * entry_price)
                profit_pct = (profit / (position * entry_price)) * 100

                trades.append({
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'exit_reason': exit_reason
                })
                position = 0

        total_value = cash + position * current_price
        portfolio_values.append(total_value)

    # 期末平仓
    if position > 0:
        final_price = df['Close'].iloc[-1]
        revenue = position * final_price * 0.998
        cash += revenue
        profit = revenue - (position * entry_price)
        profit_pct = (profit / (position * entry_price)) * 100

        trades.append({
            'entry_date': entry_date,
            'exit_date': df.index[-1],
            'profit': profit,
            'profit_pct': profit_pct,
            'exit_reason': 'EOD'
        })

    final_value = cash
    total_return = (final_value - 100000) / 100000 * 100

    # 计算指标
    portfolio_series = pd.Series(portfolio_values)
    cummax = portfolio_series.cummax()
    drawdown = (portfolio_series - cummax) / cummax * 100
    max_drawdown = drawdown.min()

    returns = portfolio_series.pct_change().dropna()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

    if trades:
        win_trades = [t for t in trades if t['profit'] > 0]
        win_rate = len(win_trades) / len(trades) if trades else 0
    else:
        win_rate = 0

    return {
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'win_rate': win_rate,
        'num_trades': len(trades),
        'final_value': final_value,
        'trades': trades
    }

def main():
    """测试2016年单独表现（完全未见数据）"""

    test_params = {
        'MSFT': {'buy': 0, 'and': 15, 'or': 50},
        'AAPL': {'buy': -2, 'and': 10, 'or': 40},
        'NVDA': {'buy': 5, 'and': 15, 'or': 50},
        'TSLA': {'buy': -5, 'and': 15, 'or': 50},
    }

    print("="*80)
    print("2016年单独测试（完全未见数据，0%数据泄漏）")
    print("="*80)
    print("\n目的: 验证真正的跨时期泛化能力\n")

    results = []

    for symbol, params in test_params.items():
        print(f"\n{'='*70}")
        print(f"测试 {symbol} - 仅2016年")
        print(f"{'='*70}")
        print(f"参数: buy<{params['buy']}, and>{params['and']}, or>{params['or']}")

        loader = DataLoader(db_config)
        prices = loader.load_ohlcv(symbol, '2012-01-01', '2016-12-31')
        sentiment = load_sentiment_index(symbol, smoothing=3)

        common_dates = prices.index.intersection(sentiment.index)
        prices = prices.loc[common_dates]
        sentiment = sentiment.loc[common_dates]

        # 只测试2016年
        prices_test = prices.loc['2016-01-01':'2016-12-31']
        sentiment_test = sentiment.loc['2016-01-01':'2016-12-31']

        print(f"2016年数据点: {len(prices_test)}")

        result = backtest_strategy(
            prices_test,
            sentiment_test,
            params['buy'],
            params['and'],
            params['or']
        )

        if result:
            print(f"\n回测结果:")
            print(f"总收益: {result['total_return']:.2f}%")
            print(f"Sharpe比率: {result['sharpe_ratio']:.4f}")
            print(f"最大回撤: {result['max_drawdown']:.2f}%")
            print(f"胜率: {result['win_rate']*100:.1f}%")
            print(f"交易次数: {result['num_trades']}")
            print(f"是否盈利: {'✅ 是' if result['total_return'] > 0 else '❌ 否'}")

            results.append({
                'symbol': symbol,
                'return': result['total_return'],
                'sharpe': result['sharpe_ratio'],
                'max_dd': result['max_drawdown'],
                'win_rate': result['win_rate'],
                'trades': result['num_trades'],
                'profitable': result['total_return'] > 0
            })
        else:
            print("❌ 数据不足")
            results.append({
                'symbol': symbol,
                'return': 0,
                'sharpe': 0,
                'max_dd': 0,
                'win_rate': 0,
                'trades': 0,
                'profitable': False
            })

    print(f"\n{'='*80}")
    print("2016年测试总结（完全未见数据）")
    print(f"{'='*80}\n")

    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

    # 保存
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'../results/test_2016_only_{timestamp}.csv'
    df_results.to_csv(filename, index=False)
    print(f"\n✅ 结果已保存: {filename}")

    return df_results

if __name__ == "__main__":
    main()
