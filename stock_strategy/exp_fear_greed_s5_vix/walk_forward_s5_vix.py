"""
Walk-Forward 分析 - 使用 fear_greed_index_s5_vix 表
四因子模型: 原三因子 + VIX
平滑参数: Smoothing=5
美股七姐妹: TSLA, AAPL, MSFT, GOOGL, AMZN, NVDA, META
"""

import sys
sys.path.insert(0, '/Users/sc2025/Desktop/Claude/Quantrade/src')

import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime
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


def load_sentiment_s5_vix(symbol):
    """从 fear_greed_index_s5_vix 表加载情绪数据 (四因子+VIX)"""
    conn = psycopg2.connect(**db_config)
    query = """
        SELECT date, smoothed_index
        FROM fear_greed_index_s5_vix
        WHERE symbol = %s
        ORDER BY date
    """
    df = pd.read_sql(query, conn, params=(symbol,))
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.set_index('date')
    conn.close()
    return df


def load_price(symbol):
    """加载价格数据"""
    loader = DataLoader(db_config)
    price_df = loader.load_ohlcv(symbol, start_date="2011-01-01")
    loader.close()
    return price_df


def run_backtest(price_data, sentiment_data, buy_threshold, and_sell_threshold, or_threshold,
                 initial_cash, initial_position=0, initial_entry_price=0):
    """运行回测"""
    df = price_data.copy()
    df['sentiment'] = sentiment_data['smoothed_index'].reindex(df.index)
    df['MA50'] = df['Close'].rolling(50).mean()
    df = df.dropna()

    if len(df) < 10:
        return initial_cash, [], initial_cash, 0, 0

    cash = initial_cash
    position = initial_position
    entry_price = initial_entry_price
    trades = []

    for i in range(len(df)):
        current_date = df.index[i]
        current_price = df['Close'].iloc[i]
        current_sentiment = df['sentiment'].iloc[i]
        current_ma50 = df['MA50'].iloc[i]

        # 卖出信号
        if position > 0:
            sell_signal = False
            sell_reason = ""

            if current_sentiment > or_threshold:
                sell_signal = True
                sell_reason = f"OR: {current_sentiment:.1f} > {or_threshold}"
            elif current_sentiment > and_sell_threshold and current_price < current_ma50:
                sell_signal = True
                sell_reason = f"AND: {current_sentiment:.1f} > {and_sell_threshold}"

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
                        'reason': f"{current_sentiment:.1f} < {buy_threshold}"
                    })

    final_value = cash + position * df['Close'].iloc[-1]
    return final_value, trades, cash, position, entry_price


def grid_search(price_data, sentiment_data):
    """网格搜索最优参数"""
    best_return = -float('inf')
    best_params = None
    best_trades = 0

    for buy_t in BUY_THRESHOLDS:
        for and_t in AND_SELL_THRESHOLDS:
            for or_t in OR_THRESHOLDS:
                final_value, trades, _, _, _ = run_backtest(
                    price_data, sentiment_data, buy_t, and_t, or_t,
                    initial_cash=INITIAL_CAPITAL
                )
                ret = (final_value / INITIAL_CAPITAL - 1) * 100

                if ret > best_return:
                    best_return = ret
                    best_params = (buy_t, and_t, or_t)
                    best_trades = len(trades)

    return best_params, best_return, best_trades


def analyze_symbol(symbol):
    """分析单个股票"""
    print(f"\n{'='*70}")
    print(f"  {symbol} Walk-Forward 分析 (S5+VIX 四因子)")
    print(f"{'='*70}")

    # 加载数据
    price_df = load_price(symbol)
    sentiment_df = load_sentiment_s5_vix(symbol)

    print(f"  价格数据: {price_df.index.min().date()} ~ {price_df.index.max().date()}")
    print(f"  情绪数据: {sentiment_df.index.min().date()} ~ {sentiment_df.index.max().date()}")

    results = []

    # 连续回测状态
    cash = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    all_trades = []

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
        best_params, train_return, train_trades = grid_search(train_price, train_sentiment)
        buy_t, and_t, or_t = best_params

        # 测试期参数（阈值放宽）
        test_buy_t = buy_t * THRESHOLD_RELAX_FACTOR

        # 测试期开始资产
        test_start_value = cash + position * test_price['Close'].iloc[0] if len(test_price) > 0 else cash

        # 测试期回测
        final_value, trades, cash, position, entry_price = run_backtest(
            test_price, test_sentiment, test_buy_t, and_t, or_t,
            initial_cash=cash, initial_position=position, initial_entry_price=entry_price
        )

        test_return = (final_value / test_start_value - 1) * 100 if test_start_value > 0 else 0

        # 记录结果
        results.append({
            'window': window_name,
            'train_buy': buy_t,
            'train_and': and_t,
            'train_or': or_t,
            'train_return': train_return,
            'train_trades': train_trades,
            'test_buy': test_buy_t,
            'test_return': test_return,
            'test_trades': len(trades),
            'end_value': final_value
        })

        # 记录交易
        for t in trades:
            t['window'] = window_name
            t['symbol'] = symbol
            all_trades.append(t)

        # 打印结果
        print(f"\n  {window_name}: Train {train_start[:4]}-{train_end[:4]} → Test {test_start[:4]}")
        print(f"    训练参数: Buy<{buy_t}, AND>{and_t}, OR>{or_t} → +{train_return:.1f}%")
        print(f"    测试参数: Buy<{test_buy_t:.0f}, AND>{and_t}, OR>{or_t} → {test_return:+.1f}%")
        if trades:
            for t in trades:
                if t['type'] == 'BUY':
                    print(f"      {t['date'].strftime('%Y-%m-%d')} BUY  @ ${t['price']:.2f} | Sentiment: {t['sentiment']:.1f}")
                else:
                    print(f"      {t['date'].strftime('%Y-%m-%d')} SELL @ ${t['price']:.2f} | {t['profit_pct']:+.1f}%")

    # 计算总收益
    total_return = (final_value / INITIAL_CAPITAL - 1) * 100
    print(f"\n  总收益: ${INITIAL_CAPITAL:,} → ${final_value:,.0f} ({total_return:+.1f}%)")

    return {
        'symbol': symbol,
        'total_return': total_return,
        'final_value': final_value,
        'results': results,
        'trades': all_trades,
        'recommended_2026': results[-1] if results else None
    }


def main():
    """主函数"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("="*70)
    print("Walk-Forward 分析 - fear_greed_index_s5_vix (四因子+VIX)")
    print("="*70)

    all_results = []
    all_trades = []
    recommendations = []

    for symbol in SYMBOLS:
        result = analyze_symbol(symbol)
        all_results.append(result)
        all_trades.extend(result['trades'])

        if result['recommended_2026']:
            r = result['recommended_2026']
            recommendations.append({
                'symbol': symbol,
                'buy_threshold': r['train_buy'],
                'and_threshold': r['train_and'],
                'or_threshold': r['train_or'],
                'test_buy_2026': r['train_buy'] * THRESHOLD_RELAX_FACTOR,
                'train_return': r['train_return'],
                'test_return_2025': r['test_return'],
                'total_return': result['total_return']
            })

    # 保存结果
    # 汇总
    summary_df = pd.DataFrame([{
        'symbol': r['symbol'],
        'total_return': r['total_return'],
        'final_value': r['final_value']
    } for r in all_results])
    summary_df = summary_df.sort_values('total_return', ascending=False)
    summary_df.to_csv(f'summary_s5_vix_{timestamp}.csv', index=False)

    # 交易记录
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_df['date'] = trades_df['date'].dt.strftime('%Y-%m-%d')
        trades_df.to_csv(f'trades_s5_vix_{timestamp}.csv', index=False)

    # 2026推荐参数
    rec_df = pd.DataFrame(recommendations)
    rec_df = rec_df.sort_values('total_return', ascending=False)
    rec_df.to_csv(f'recommendations_2026_s5_vix_{timestamp}.csv', index=False)

    # 打印2026推荐参数
    print("\n" + "="*70)
    print("2026年推荐参数 (S5+VIX 四因子, 基于2021-2024训练)")
    print("="*70)
    print(f"\n{'股票':<8} {'Buy':<8} {'AND':<8} {'OR':<8} {'2025测试':<12} {'总收益':<12}")
    print("-"*70)
    for _, row in rec_df.iterrows():
        print(f"{row['symbol']:<8} <{row['buy_threshold']:<5} >{row['and_threshold']:<5} >{row['or_threshold']:<5} {row['test_return_2025']:>+10.1f}% {row['total_return']:>+10.1f}%")

    # 最终排名
    print("\n" + "="*70)
    print("最终排名 (连续回测 2020-2025, S5+VIX)")
    print("="*70)
    print(f"\n{'排名':<4} {'股票':<8} {'最终资产':>14} {'总收益':>12}")
    print("-"*50)
    for i, (_, row) in enumerate(summary_df.iterrows(), 1):
        print(f"{i:<4} {row['symbol']:<8} ${row['final_value']:>12,.0f} {row['total_return']:>+10.1f}%")

    print(f"\n输出文件:")
    print(f"  - summary_s5_vix_{timestamp}.csv")
    print(f"  - trades_s5_vix_{timestamp}.csv")
    print(f"  - recommendations_2026_s5_vix_{timestamp}.csv")


if __name__ == "__main__":
    main()
