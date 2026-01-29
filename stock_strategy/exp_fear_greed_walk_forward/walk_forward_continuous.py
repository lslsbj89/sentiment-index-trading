"""
Walk-Forward 连续回测
- 资金和持仓跨窗口延续
- 每个窗口使用该窗口训练期的最优参数
- 记录完整的买卖交易
"""

import sys
sys.path.insert(0, '/Users/sc2025/Desktop/Claude/Quantrade/src')

import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime
import matplotlib.pyplot as plt
import os
from data_loader import DataLoader

# 数据库配置
db_config = {
    "host": "localhost",
    "port": 5432,
    "database": "crypto_fear_greed_2",
    "user": "sc2025",
    "password": ""
}

# 参数
INITIAL_CAPITAL = 100000
POSITION_PCT = 0.8
COMMISSION = 0.001
SLIPPAGE = 0.001

# 网格搜索参数
BUY_THRESHOLDS = [-30, -25, -20, -15, -10, -5, 0, 5]
AND_SELL_THRESHOLDS = [5, 10, 15, 20, 25]
OR_THRESHOLDS = [25, 30, 35, 40, 45, 50, 55, 60, 65]

# 阈值放宽因子
THRESHOLD_RELAX_FACTOR = 0.8

# Walk-Forward 窗口
WINDOWS = [
    {"name": "W2020", "train": ("2016-01-01", "2019-12-31"), "test": ("2020-01-01", "2020-12-31")},
    {"name": "W2021", "train": ("2017-01-01", "2020-12-31"), "test": ("2021-01-01", "2021-12-31")},
    {"name": "W2022", "train": ("2018-01-01", "2021-12-31"), "test": ("2022-01-01", "2022-12-31")},
    {"name": "W2023", "train": ("2019-01-01", "2022-12-31"), "test": ("2023-01-01", "2023-12-31")},
    {"name": "W2024", "train": ("2020-01-01", "2023-12-31"), "test": ("2024-01-01", "2024-12-31")},
    {"name": "W2025", "train": ("2021-01-01", "2024-12-31"), "test": ("2025-01-01", "2025-12-31")},
]

SYMBOLS = ["TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"]


def load_data(symbol):
    """加载价格和情绪数据"""
    # 价格数据
    loader = DataLoader(db_config)
    price_df = loader.load_ohlcv(symbol, start_date="2011-01-01")
    loader.close()

    # 情绪数据
    conn = psycopg2.connect(**db_config)
    sentiment_query = """
        SELECT date, smoothed_index
        FROM fear_greed_index
        WHERE symbol = %s
        ORDER BY date
    """
    sentiment_df = pd.read_sql(sentiment_query, conn, params=(symbol,))
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'], utc=True)
    sentiment_df = sentiment_df.set_index('date')
    conn.close()

    return price_df, sentiment_df


def run_backtest(price_data, sentiment_data, buy_threshold, and_sell_threshold, or_threshold,
                 initial_cash, initial_position=0, initial_entry_price=0):
    """
    运行回测

    Returns:
        final_value, trades, end_cash, end_position, end_entry_price, daily_values
    """
    df = price_data.copy()
    df['sentiment'] = sentiment_data['smoothed_index'].reindex(df.index)
    df['MA50'] = df['Close'].rolling(50).mean()
    df = df.dropna()

    if len(df) < 10:
        return initial_cash, [], initial_cash, 0, 0, []

    cash = initial_cash
    position = initial_position
    entry_price = initial_entry_price
    trades = []
    daily_values = []

    for i in range(len(df)):
        current_date = df.index[i]
        current_price = df['Close'].iloc[i]
        current_sentiment = df['sentiment'].iloc[i]
        current_ma50 = df['MA50'].iloc[i]

        # 记录每日价值
        total_value = cash + position * current_price
        daily_values.append({'date': current_date, 'value': total_value})

        # 卖出信号
        if position > 0:
            sell_signal = False
            sell_reason = ""

            if current_sentiment > or_threshold:
                sell_signal = True
                sell_reason = f"OR: sentiment {current_sentiment:.1f} > {or_threshold}"
            elif current_sentiment > and_sell_threshold and current_price < current_ma50:
                sell_signal = True
                sell_reason = f"AND: sentiment {current_sentiment:.1f} > {and_sell_threshold} & price < MA50"

            if sell_signal:
                sell_price = current_price * (1 - SLIPPAGE) * (1 - COMMISSION)
                profit_pct = (sell_price - entry_price) / entry_price * 100
                cash += position * sell_price
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

        # 买入信号
        elif position == 0:
            if current_sentiment < buy_threshold:
                buy_price = current_price * (1 + SLIPPAGE) * (1 + COMMISSION)
                target_value = (cash + position * current_price) * POSITION_PCT
                shares = int(target_value / buy_price)

                if shares > 0 and cash >= shares * buy_price:
                    cash -= shares * buy_price
                    position = shares
                    entry_price = buy_price
                    trades.append({
                        'type': 'BUY',
                        'date': current_date,
                        'price': current_price,
                        'shares': shares,
                        'sentiment': current_sentiment,
                        'reason': f"sentiment {current_sentiment:.1f} < {buy_threshold}",
                        'profit_pct': 0
                    })

    final_value = cash + position * df['Close'].iloc[-1]
    return final_value, trades, cash, position, entry_price, daily_values


def grid_search_train(price_data, sentiment_data):
    """训练期网格搜索，返回最优参数"""
    best_return = -float('inf')
    best_params = None

    for buy_t in BUY_THRESHOLDS:
        for and_t in AND_SELL_THRESHOLDS:
            for or_t in OR_THRESHOLDS:
                final_value, trades, _, _, _, _ = run_backtest(
                    price_data, sentiment_data, buy_t, and_t, or_t,
                    initial_cash=INITIAL_CAPITAL
                )
                ret = (final_value / INITIAL_CAPITAL - 1) * 100

                if ret > best_return:
                    best_return = ret
                    best_params = (buy_t, and_t, or_t)

    return best_params, best_return


def run_continuous_backtest(symbol):
    """
    连续回测：资金和持仓跨窗口延续
    """
    print(f"\n{'='*70}")
    print(f"  {symbol} 连续回测")
    print(f"{'='*70}")

    # 加载数据
    price_df, sentiment_df = load_data(symbol)

    # 初始状态
    cash = INITIAL_CAPITAL
    position = 0
    entry_price = 0

    all_trades = []
    all_daily_values = []
    window_results = []

    for window in WINDOWS:
        window_name = window['name']
        train_start, train_end = window['train']
        test_start, test_end = window['test']

        # 训练期数据
        train_price = price_df[train_start:train_end]
        train_sentiment = sentiment_df[train_start:train_end]

        # 测试期数据
        test_price = price_df[test_start:test_end]
        test_sentiment = sentiment_df[test_start:test_end]

        if len(train_price) < 100 or len(test_price) < 10:
            print(f"  {window_name}: 数据不足，跳过")
            continue

        # 训练期网格搜索
        best_params, train_return = grid_search_train(train_price, train_sentiment)
        buy_t, and_t, or_t = best_params

        # 测试期参数（阈值放宽）
        test_buy_t = buy_t * THRESHOLD_RELAX_FACTOR

        # 记录测试期开始时的资产
        test_start_value = cash + position * test_price['Close'].iloc[0] if len(test_price) > 0 else cash

        # 测试期回测（使用当前资金和持仓）
        final_value, trades, cash, position, entry_price, daily_values = run_backtest(
            test_price, test_sentiment, test_buy_t, and_t, or_t,
            initial_cash=cash, initial_position=position, initial_entry_price=entry_price
        )

        # 计算测试期收益
        test_return = (final_value / test_start_value - 1) * 100 if test_start_value > 0 else 0

        # 记录交易
        for trade in trades:
            trade['window'] = window_name
            trade['symbol'] = symbol
            all_trades.append(trade)

        # 记录每日价值
        all_daily_values.extend(daily_values)

        # 记录窗口结果
        window_results.append({
            'window': window_name,
            'train_params': f"Buy<{buy_t}, AND>{and_t}, OR>{or_t}",
            'test_params': f"Buy<{test_buy_t:.0f}, AND>{and_t}, OR>{or_t}",
            'train_return': train_return,
            'test_return': test_return,
            'start_value': test_start_value,
            'end_value': final_value,
            'trades': len(trades),
            'position': position,
            'cash': cash
        })

        # 打印窗口结果
        print(f"\n  {window_name}: Train {train_start[:4]}-{train_end[:4]} → Test {test_start[:4]}")
        print(f"    训练参数: Buy<{buy_t}, AND>{and_t}, OR>{or_t} (收益: +{train_return:.1f}%)")
        print(f"    测试参数: Buy<{test_buy_t:.0f}, AND>{and_t}, OR>{or_t}")
        print(f"    期初资产: ${test_start_value:,.0f} → 期末: ${final_value:,.0f} ({test_return:+.1f}%)")
        print(f"    交易次数: {len(trades)}, 当前持仓: {position}股")

        if trades:
            print(f"    交易记录:")
            for t in trades:
                if t['type'] == 'BUY':
                    print(f"      {t['date'].strftime('%Y-%m-%d')} BUY  {t['shares']}股 @ ${t['price']:.2f} | Sentiment: {t['sentiment']:.1f}")
                else:
                    print(f"      {t['date'].strftime('%Y-%m-%d')} SELL {t['shares']}股 @ ${t['price']:.2f} | Sentiment: {t['sentiment']:.1f} | {t['profit_pct']:+.1f}%")

    # 计算总收益
    total_return = (final_value / INITIAL_CAPITAL - 1) * 100

    print(f"\n  {'='*50}")
    print(f"  总结: ${INITIAL_CAPITAL:,} → ${final_value:,.0f} ({total_return:+.1f}%)")
    print(f"  总交易次数: {len(all_trades)}")
    if position > 0:
        print(f"  当前持仓: {position}股 @ ${entry_price:.2f}")
    print(f"  {'='*50}")

    return {
        'symbol': symbol,
        'initial_capital': INITIAL_CAPITAL,
        'final_value': final_value,
        'total_return': total_return,
        'total_trades': len(all_trades),
        'end_position': position,
        'end_cash': cash,
        'trades': all_trades,
        'daily_values': all_daily_values,
        'window_results': window_results
    }


def main():
    """主函数"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"continuous_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print("="*70)
    print("Walk-Forward 连续回测 - 资金持仓跨窗口延续")
    print("="*70)

    all_results = []
    all_trades = []

    for symbol in SYMBOLS:
        result = run_continuous_backtest(symbol)
        all_results.append(result)
        all_trades.extend(result['trades'])

    # 保存汇总结果
    summary_df = pd.DataFrame([{
        'symbol': r['symbol'],
        'initial': r['initial_capital'],
        'final': r['final_value'],
        'return_pct': r['total_return'],
        'trades': r['total_trades'],
        'end_position': r['end_position']
    } for r in all_results])
    summary_df = summary_df.sort_values('return_pct', ascending=False)
    summary_df.to_csv(f"{output_dir}/summary.csv", index=False)

    # 保存所有交易记录
    trades_df = pd.DataFrame(all_trades)
    if len(trades_df) > 0:
        trades_df['date'] = trades_df['date'].dt.strftime('%Y-%m-%d')
        trades_df.to_csv(f"{output_dir}/all_trades.csv", index=False)

    # 打印最终排名
    print("\n" + "="*70)
    print("最终排名 (连续回测 2020-2025)")
    print("="*70)
    print(f"\n{'排名':<4} {'股票':<8} {'初始资金':>12} {'最终资产':>14} {'总收益':>12} {'交易次数':>8}")
    print("-"*70)

    for i, (_, row) in enumerate(summary_df.iterrows(), 1):
        print(f"{i:<4} {row['symbol']:<8} ${row['initial']:>10,} ${row['final']:>12,.0f} {row['return_pct']:>+10.1f}% {row['trades']:>8}")

    print(f"\n输出目录: {output_dir}/")
    print(f"  - summary.csv: 汇总结果")
    print(f"  - all_trades.csv: 所有交易记录")


if __name__ == "__main__":
    main()
