#!/usr/bin/env python3
"""
详细交易执行分析

展示使用最优参数时的每笔交易细节：
- 买入时间、价格、指数、MA50
- 卖出时间、价格、指数、MA50、退出原因
- 持仓天数、收益率、收益金额
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime
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
    """
    回测并返回详细的交易记录
    """
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
                exit_reason = 'OR卖出 (指数>阈值)'
            elif current_idx > sell_and_threshold and current_price < current_ma50:
                sell_signal = True
                exit_reason = 'AND卖出 (指数>阈值且破MA50)'
            elif holding_days >= 60:
                sell_signal = True
                exit_reason = '超时卖出 (≥60天)'

            if sell_signal:
                revenue = position * current_price * 0.998
                cash += revenue

                profit = revenue - (position * entry_price)
                profit_pct = (profit / (position * entry_price)) * 100

                # 记录交易
                trades.append({
                    'entry_date': entry_date.strftime('%Y-%m-%d'),
                    'entry_price': round(entry_price, 2),
                    'entry_idx': round(entry_idx, 2),
                    'entry_ma50': round(entry_ma50, 2),
                    'exit_date': current_date.strftime('%Y-%m-%d'),
                    'exit_price': round(current_price * 0.998, 2),
                    'exit_idx': round(current_idx, 2),
                    'exit_ma50': round(current_ma50, 2),
                    'exit_reason': exit_reason,
                    'holding_days': holding_days,
                    'shares': position,
                    'cost': round(position * entry_price, 2),
                    'revenue': round(revenue, 2),
                    'profit': round(profit, 2),
                    'profit_pct': round(profit_pct, 2)
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
            'entry_date': entry_date.strftime('%Y-%m-%d'),
            'entry_price': round(entry_price, 2),
            'entry_idx': round(entry_idx, 2),
            'entry_ma50': round(entry_ma50, 2),
            'exit_date': df.index[-1].strftime('%Y-%m-%d'),
            'exit_price': round(final_price * 0.998, 2),
            'exit_idx': round(final_idx, 2),
            'exit_ma50': round(final_ma50, 2),
            'exit_reason': '期末强制平仓',
            'holding_days': holding_days,
            'shares': position,
            'cost': round(position * entry_price, 2),
            'revenue': round(revenue, 2),
            'profit': round(profit, 2),
            'profit_pct': round(profit_pct, 2)
        })

    final_value = cash
    total_return = (final_value - 100000) / 100000 * 100

    return pd.DataFrame(trades), total_return

def print_trade_details(trades_df, symbol, params):
    """打印交易详情"""
    print(f"\n{'='*100}")
    print(f"{symbol} 交易详细记录")
    print(f"{'='*100}")
    print(f"\n使用参数: buy_threshold<{params['buy']}, and_threshold>{params['and']}, or_threshold>{params['or']}")
    print(f"总交易次数: {len(trades_df)}")

    # 统计
    win_trades = trades_df[trades_df['profit'] > 0]
    loss_trades = trades_df[trades_df['profit'] <= 0]

    print(f"\n盈利交易: {len(win_trades)} 笔 ({len(win_trades)/len(trades_df)*100:.1f}%)")
    print(f"亏损交易: {len(loss_trades)} 笔 ({len(loss_trades)/len(trades_df)*100:.1f}%)")
    print(f"平均盈利: {win_trades['profit_pct'].mean():.2f}%" if len(win_trades) > 0 else "平均盈利: N/A")
    print(f"平均亏损: {loss_trades['profit_pct'].mean():.2f}%" if len(loss_trades) > 0 else "平均亏损: N/A")

    # 退出原因统计
    print(f"\n退出原因分布:")
    exit_reasons = trades_df['exit_reason'].value_counts()
    for reason, count in exit_reasons.items():
        print(f"  {reason}: {count} 笔 ({count/len(trades_df)*100:.1f}%)")

    # 详细交易记录
    print(f"\n{'='*100}")
    print("详细交易记录:")
    print(f"{'='*100}\n")

    for i, trade in trades_df.iterrows():
        print(f"【交易 #{i+1}】{'='*85}")
        print(f"\n  买入:")
        print(f"    日期: {trade['entry_date']}")
        print(f"    价格: ${trade['entry_price']:.2f}")
        print(f"    指数: {trade['entry_idx']:.2f} {'(恐慌<' + str(params['buy']) + ')' if trade['entry_idx'] < params['buy'] else ''}")
        print(f"    MA50: ${trade['entry_ma50']:.2f}")
        print(f"    股数: {trade['shares']} 股")
        print(f"    成本: ${trade['cost']:,.2f}")

        print(f"\n  卖出:")
        print(f"    日期: {trade['exit_date']}")
        print(f"    价格: ${trade['exit_price']:.2f}")
        print(f"    指数: {trade['exit_idx']:.2f}")
        print(f"    MA50: ${trade['exit_ma50']:.2f}")
        print(f"    原因: {trade['exit_reason']}")

        print(f"\n  结果:")
        print(f"    持仓天数: {trade['holding_days']} 天")
        print(f"    卖出收入: ${trade['revenue']:,.2f}")
        profit_symbol = '✅' if trade['profit'] > 0 else '❌'
        print(f"    盈亏金额: ${trade['profit']:,.2f} {profit_symbol}")
        print(f"    盈亏比例: {trade['profit_pct']:.2f}% {profit_symbol}")
        print()

def analyze_symbol(symbol, smoothing=3, params=None):
    """分析单个股票"""
    print(f"\n{'='*100}")
    print(f"加载 {symbol} 数据...")
    print(f"{'='*100}")

    loader = DataLoader(db_config)
    prices = loader.load_ohlcv(symbol, '2020-01-01', '2025-12-31')
    sentiment = load_sentiment_index(symbol, smoothing)

    # 对齐数据
    common_dates = prices.index.intersection(sentiment.index)
    test_price = prices.loc[common_dates].loc['2021-01-01':]
    test_sentiment = sentiment.loc[common_dates].loc['2021-01-01':]

    print(f"测试期间: {test_price.index[0].strftime('%Y-%m-%d')} 至 {test_price.index[-1].strftime('%Y-%m-%d')}")
    print(f"交易日数: {len(test_price)} 天")

    # 回测
    trades_df, total_return = backtest_with_details(
        test_price, test_sentiment,
        params['buy'], params['and'], params['or']
    )

    if trades_df is None:
        print(f"❌ {symbol} 数据不足")
        return None

    # 打印详情
    print_trade_details(trades_df, symbol, params)

    # 总结
    print(f"\n{'='*100}")
    print("回测总结")
    print(f"{'='*100}")
    print(f"\n初始资金: $100,000.00")
    print(f"最终资金: ${100000 * (1 + total_return/100):,.2f}")
    print(f"总收益率: {total_return:.2f}%")
    print(f"总交易次数: {len(trades_df)}")
    print(f"总盈利: ${trades_df[trades_df['profit'] > 0]['profit'].sum():,.2f}")
    print(f"总亏损: ${trades_df[trades_df['profit'] < 0]['profit'].sum():,.2f}")

    return trades_df

def main():
    print("\n" + "="*100)
    print("详细交易执行分析 - Smoothing=3 最优参数")
    print("="*100)

    # S3 最优参数
    OPTIMAL_PARAMS = {
        'NVDA': {'buy': 10, 'and': 30, 'or': 70},
        'TSLA': {'buy': -10, 'and': 25, 'or': 50},
        'GOOGL': {'buy': 10, 'and': 30, 'or': 70},
        'AAPL': {'buy': -10, 'and': 15, 'or': 40},
    }

    # 让用户选择股票
    print("\n可选股票:")
    for i, symbol in enumerate(OPTIMAL_PARAMS.keys(), 1):
        print(f"  {i}. {symbol}")

    # 默认分析NVDA
    symbols_to_analyze = ['NVDA']  # 可以修改为其他股票

    all_trades = {}

    for symbol in symbols_to_analyze:
        if symbol in OPTIMAL_PARAMS:
            trades_df = analyze_symbol(symbol, smoothing=3, params=OPTIMAL_PARAMS[symbol])
            if trades_df is not None:
                all_trades[symbol] = trades_df

                # 保存到CSV
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'detailed_trades_{symbol}_s3_{timestamp}.csv'
                trades_df.to_csv(filename, index=False)
                print(f"\n✅ 交易记录已保存: {filename}")

    print(f"\n{'='*100}")
    print("✅ 分析完成！")
    print(f"{'='*100}\n")

if __name__ == "__main__":
    main()
