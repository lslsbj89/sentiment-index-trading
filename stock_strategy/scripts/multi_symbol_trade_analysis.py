#!/usr/bin/env python3
"""
多股票交易详细分析与对比

同时分析多个股票的交易执行情况，生成对比图表
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime
from data_loader import DataLoader
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
          AND date >= '2020-01-01'
          AND date <= '2025-12-31'
        ORDER BY date
    """
    df = pd.read_sql(query, conn, parse_dates=['date'])
    conn.close()
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df.set_index('date', inplace=True)
    return df

def backtest_with_details(prices, sentiment, buy_threshold, sell_and_threshold, sell_or_threshold):
    """回测并返回详细的交易记录"""
    df = prices.copy()
    df['idx'] = sentiment['smoothed_index']
    df['MA50'] = df['Close'].rolling(50).mean()
    df = df.dropna()

    if len(df) < 100:
        return None, None

    cash = 100000
    position = 0
    entry_price = 0
    entry_date = None
    entry_idx = 0
    entry_ma50 = 0
    holding_days = 0

    trades = []
    portfolio_values = []

    for i in range(len(df)):
        current_date = df.index[i]
        current_price = df['Close'].iloc[i]
        current_idx = df['idx'].iloc[i]
        current_ma50 = df['MA50'].iloc[i]

        # 买入逻辑
        if position == 0 and current_idx < buy_threshold:
            available = cash * 0.8
            shares = int(available / (current_price * 1.002))
            if shares > 0:
                cost = shares * current_price * 1.002
                cash -= cost
                position = shares
                entry_price = current_price * 1.002
                entry_date = current_date
                entry_idx = current_idx
                entry_ma50 = current_ma50
                holding_days = 0

        # 持仓逻辑
        elif position > 0:
            holding_days += 1

            # 判断卖出条件
            sell_signal = False
            exit_reason = None

            if current_idx > sell_or_threshold:
                sell_signal = True
                exit_reason = 'OR卖出'
            elif current_idx > sell_and_threshold and current_price < current_ma50:
                sell_signal = True
                exit_reason = 'AND卖出'
            elif holding_days >= 60:
                sell_signal = True
                exit_reason = '超时卖出'

            if sell_signal:
                revenue = position * current_price * 0.998
                cash += revenue

                profit = revenue - (position * entry_price)
                profit_pct = (profit / (position * entry_price)) * 100

                # 记录交易
                trades.append({
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'entry_idx': entry_idx,
                    'entry_ma50': entry_ma50,
                    'exit_date': current_date,
                    'exit_price': current_price * 0.998,
                    'exit_idx': current_idx,
                    'exit_ma50': current_ma50,
                    'exit_reason': exit_reason,
                    'holding_days': holding_days,
                    'shares': position,
                    'cost': position * entry_price,
                    'revenue': revenue,
                    'profit': profit,
                    'profit_pct': profit_pct
                })

                position = 0

        total_value = cash + position * current_price
        portfolio_values.append(total_value)

    # 强制平仓
    if position > 0:
        final_price = df['Close'].iloc[-1]
        final_idx = df['idx'].iloc[-1]
        final_ma50 = df['MA50'].iloc[-1]
        revenue = position * final_price * 0.998
        cash += revenue

        profit = revenue - (position * entry_price)
        profit_pct = (profit / (position * entry_price)) * 100

        trades.append({
            'entry_date': entry_date,
            'entry_price': entry_price,
            'entry_idx': entry_idx,
            'entry_ma50': entry_ma50,
            'exit_date': df.index[-1],
            'exit_price': final_price * 0.998,
            'exit_idx': final_idx,
            'exit_ma50': final_ma50,
            'exit_reason': '期末平仓',
            'holding_days': holding_days,
            'shares': position,
            'cost': position * entry_price,
            'revenue': revenue,
            'profit': profit,
            'profit_pct': profit_pct
        })

    final_value = cash
    total_return = (final_value - 100000) / 100000 * 100

    return pd.DataFrame(trades), total_return

def analyze_symbol(symbol, smoothing=3, params=None):
    """分析单个股票"""
    loader = DataLoader(db_config)
    prices = loader.load_ohlcv(symbol, '2020-01-01', '2025-12-31')
    sentiment = load_sentiment_index(symbol, smoothing)

    # 对齐数据
    common_dates = prices.index.intersection(sentiment.index)
    test_price = prices.loc[common_dates].loc['2021-01-01':]
    test_sentiment = sentiment.loc[common_dates].loc['2021-01-01':]

    # 回测
    trades_df, total_return = backtest_with_details(
        test_price, test_sentiment,
        params['buy'], params['and'], params['or']
    )

    if trades_df is None:
        return None, None

    return trades_df, total_return

def plot_multi_symbol_comparison(all_trades, all_returns):
    """绘制多股票对比图表"""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    symbols = list(all_trades.keys())

    # 1. 胜率对比
    ax1 = fig.add_subplot(gs[0, 0])
    win_rates = []
    for symbol in symbols:
        trades = all_trades[symbol]
        win_rate = (trades['profit'] > 0).sum() / len(trades) * 100
        win_rates.append(win_rate)

    ax1.bar(symbols, win_rates, color='steelblue', alpha=0.7)
    ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50%基准线')
    ax1.set_ylabel('胜率 (%)', fontsize=11)
    ax1.set_title('各股票胜率对比', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 平均收益对比
    ax2 = fig.add_subplot(gs[0, 1])
    avg_profits = []
    for symbol in symbols:
        trades = all_trades[symbol]
        avg_profit = trades['profit_pct'].mean()
        avg_profits.append(avg_profit)

    colors = ['green' if x > 0 else 'red' for x in avg_profits]
    ax2.bar(symbols, avg_profits, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('平均收益率 (%)', fontsize=11)
    ax2.set_title('各股票平均单笔收益率', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. 交易次数对比
    ax3 = fig.add_subplot(gs[1, 0])
    trade_counts = [len(all_trades[symbol]) for symbol in symbols]
    ax3.bar(symbols, trade_counts, color='orange', alpha=0.7)
    ax3.set_ylabel('交易次数', fontsize=11)
    ax3.set_title('各股票交易次数', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. 总收益率对比
    ax4 = fig.add_subplot(gs[1, 1])
    returns = [all_returns[symbol] for symbol in symbols]
    colors = ['green' if x > 0 else 'red' for x in returns]
    ax4.bar(symbols, returns, color=colors, alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_ylabel('总收益率 (%)', fontsize=11)
    ax4.set_title('各股票总收益率 (2021-2025)', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # 5. 盈亏分布
    ax5 = fig.add_subplot(gs[2, :])
    for symbol in symbols:
        trades = all_trades[symbol]
        profits = trades['profit_pct'].values
        ax5.scatter([symbol] * len(profits), profits, alpha=0.6, s=50, label=symbol)

    ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax5.set_ylabel('单笔收益率 (%)', fontsize=11)
    ax5.set_title('各股票交易盈亏分布', fontsize=13, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'multi_symbol_comparison_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✅ 对比图表已保存: {filename}")
    plt.close()

def print_summary_table(all_trades, all_returns):
    """打印汇总表格"""
    print(f"\n{'='*120}")
    print("多股票交易汇总表")
    print(f"{'='*120}\n")

    summary_data = []

    for symbol in all_trades.keys():
        trades = all_trades[symbol]
        win_trades = trades[trades['profit'] > 0]
        loss_trades = trades[trades['profit'] <= 0]

        summary_data.append({
            'symbol': symbol,
            'total_trades': len(trades),
            'win_trades': len(win_trades),
            'loss_trades': len(loss_trades),
            'win_rate': len(win_trades) / len(trades) * 100,
            'avg_profit': win_trades['profit_pct'].mean() if len(win_trades) > 0 else 0,
            'avg_loss': loss_trades['profit_pct'].mean() if len(loss_trades) > 0 else 0,
            'total_return': all_returns[symbol]
        })

    df_summary = pd.DataFrame(summary_data)

    print(f"{'股票':<8} {'交易次数':<10} {'盈利笔数':<10} {'亏损笔数':<10} {'胜率':<10} {'平均盈利':<12} {'平均亏损':<12} {'总收益率':<12}")
    print("-" * 120)

    for _, row in df_summary.iterrows():
        print(f"{row['symbol']:<8} {row['total_trades']:<10} {row['win_trades']:<10} {row['loss_trades']:<10} "
              f"{row['win_rate']:<10.1f}% {row['avg_profit']:<12.2f}% {row['avg_loss']:<12.2f}% {row['total_return']:<12.2f}%")

    print(f"\n{'='*120}")

    # 保存汇总表
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'multi_symbol_summary_{timestamp}.csv'
    df_summary.to_csv(filename, index=False)
    print(f"\n✅ 汇总表已保存: {filename}")

def main():
    print("\n" + "="*120)
    print("多股票交易详细分析 - Smoothing=3 最优参数")
    print("="*120)

    # S3 最优参数
    OPTIMAL_PARAMS = {
        'NVDA': {'buy': 10, 'and': 30, 'or': 70},
        'TSLA': {'buy': -10, 'and': 25, 'or': 50},
        'AAPL': {'buy': -10, 'and': 15, 'or': 40},
        'META': {'buy': 0, 'and': 15, 'or': 70},
    }

    all_trades = {}
    all_returns = {}

    for symbol, params in OPTIMAL_PARAMS.items():
        print(f"\n正在分析 {symbol}...")
        try:
            trades_df, total_return = analyze_symbol(symbol, smoothing=3, params=params)
            if trades_df is not None:
                all_trades[symbol] = trades_df
                all_returns[symbol] = total_return
                print(f"✅ {symbol}: {len(trades_df)} 笔交易, 总收益 {total_return:.2f}%")
        except Exception as e:
            print(f"❌ {symbol} 分析失败: {e}")

    if all_trades:
        # 打印汇总表
        print_summary_table(all_trades, all_returns)

        # 生成对比图表
        print("\n正在生成对比图表...")
        plot_multi_symbol_comparison(all_trades, all_returns)

    print(f"\n{'='*120}")
    print("✅ 多股票分析完成！")
    print(f"{'='*120}\n")

if __name__ == "__main__":
    main()
