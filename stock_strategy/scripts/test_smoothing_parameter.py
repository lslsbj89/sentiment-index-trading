"""
Smoothing 参数对比实验

目标: 对比 smoothing = 3, 5, 7, 10 对策略的影响

测试维度:
1. 信号频率
2. 收益率
3. 夏普率
4. 最大回撤
5. 胜率
"""

import sys
import os
import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime

# Database configuration
db_config = {
    "host": "localhost",
    "port": 5432,
    "database": "crypto_fear_greed_2",
    "user": "sc2025",
    "password": ""
}

def calculate_rma(series, period):
    """计算RMA (Wilder's Moving Average)"""
    alpha = 1.0 / period
    rma = series.ewm(alpha=alpha, adjust=False).mean()
    return rma

def load_raw_index_components(symbol="TSLA", start_date="2016-01-01"):
    """加载原始指数成分"""
    conn = psycopg2.connect(**db_config)

    query = f"""
        SELECT date, pmacd, ror, money_flow, volatility, market_dom, raw_index
        FROM fear_greed_index
        WHERE symbol = '{symbol}' AND date >= '{start_date}'
        ORDER BY date
    """

    df = pd.read_sql(query, conn)
    conn.close()

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    return df

def generate_smoothed_index(raw_df, smoothing=5):
    """根据raw_index生成不同smoothing的平滑指数"""
    smoothed = calculate_rma(raw_df['raw_index'], smoothing)
    return smoothed

def load_price_data(symbol="TSLA", start_date="2016-01-01"):
    """加载价格数据"""
    conn = psycopg2.connect(**db_config)

    query = f"""
        SELECT date, open, high, low, close, volume
        FROM prices
        WHERE symbol = '{symbol}' AND date >= '{start_date}'
        ORDER BY date
    """

    df = pd.read_sql(query, conn)
    conn.close()

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    return df

def calculate_ma50(price_df):
    """计算50日移动平均"""
    return price_df['close'].rolling(window=50).mean()

def backtest_sentiment_strategy(
    price_df,
    sentiment_series,
    buy_threshold=5,
    sell_and_threshold=20,
    sell_or_threshold=40,
    position_pct=0.8,
    initial_capital=100000
):
    """
    情绪策略回测

    买入: sentiment < buy_threshold
    卖出: (sentiment > sell_and AND price < MA50) OR sentiment > sell_or
    """

    # 合并数据
    data = pd.DataFrame({
        'close': price_df['close'],
        'sentiment': sentiment_series,
        'ma50': calculate_ma50(price_df)
    }).dropna()

    # 初始化
    cash = initial_capital
    position = 0
    trades = []
    portfolio_value = []

    entry_price = 0
    entry_date = None
    entry_sentiment = 0

    for date, row in data.iterrows():
        price = row['close']
        sentiment = row['sentiment']
        ma50 = row['ma50']

        # 买入逻辑
        if position == 0 and sentiment < buy_threshold:
            shares = int((cash * position_pct) / price)
            if shares > 0:
                cost = shares * price * 1.001  # 0.1% commission
                if cost <= cash:
                    position = shares
                    cash -= cost
                    entry_price = price
                    entry_date = date
                    entry_sentiment = sentiment

        # 卖出逻辑
        elif position > 0:
            sell = False
            exit_reason = ''

            if sentiment > sell_or_threshold:
                sell = True
                exit_reason = f'idx>{sell_or_threshold}'
            elif sentiment > sell_and_threshold and price < ma50:
                sell = True
                exit_reason = f'idx>{sell_and_threshold} & <MA50'

            if sell:
                revenue = position * price * 0.999  # 0.1% commission
                cash += revenue

                profit_pct = (price - entry_price) / entry_price
                holding_days = (date - entry_date).days

                trades.append({
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'entry_sentiment': entry_sentiment,
                    'exit_date': date,
                    'exit_price': price,
                    'exit_sentiment': sentiment,
                    'profit_pct': profit_pct * 100,
                    'holding_days': holding_days,
                    'exit_reason': exit_reason
                })

                position = 0

        # 记录组合价值
        total_value = cash + (position * price if position > 0 else 0)
        portfolio_value.append({
            'date': date,
            'total_value': total_value,
            'cash': cash,
            'position_value': position * price if position > 0 else 0
        })

    # 计算指标
    portfolio_df = pd.DataFrame(portfolio_value).set_index('date')
    trades_df = pd.DataFrame(trades)

    final_value = portfolio_df['total_value'].iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital * 100

    # 计算最大回撤
    cummax = portfolio_df['total_value'].cummax()
    drawdown = (portfolio_df['total_value'] - cummax) / cummax * 100
    max_drawdown = drawdown.min()

    # 计算夏普率
    returns = portfolio_df['total_value'].pct_change().dropna()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

    # 交易统计
    num_trades = len(trades_df)
    win_rate = (trades_df['profit_pct'] > 0).sum() / num_trades * 100 if num_trades > 0 else 0

    metrics = {
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'final_value': final_value
    }

    return metrics, trades_df, portfolio_df

def run_smoothing_comparison(symbol="TSLA", test_period="2021-2025"):
    """运行不同smoothing参数的对比实验"""

    print(f"\n{'='*80}")
    print(f"Smoothing 参数对比实验 - {symbol}")
    print(f"测试周期: {test_period}")
    print(f"{'='*80}\n")

    # 加载数据
    print("加载数据...")
    if test_period == "2021-2025":
        start_date = "2020-01-01"  # 提前加载用于计算MA50
        test_start = "2021-01-01"
    else:
        start_date = "2015-01-01"
        test_start = "2016-01-01"

    raw_df = load_raw_index_components(symbol, start_date)
    price_df = load_price_data(symbol, start_date)

    # 对齐数据
    common_dates = raw_df.index.intersection(price_df.index)
    raw_df = raw_df.loc[common_dates]
    price_df = price_df.loc[common_dates]

    # 测试不同的smoothing参数
    smoothing_values = [3, 5, 7, 10]

    # 新版指数参数
    params = {
        'buy_threshold': 5,
        'sell_and_threshold': 20,
        'sell_or_threshold': 40,
        'position_pct': 0.8,
        'initial_capital': 100000
    }

    results = []

    for smoothing in smoothing_values:
        print(f"\n{'='*60}")
        print(f"测试 Smoothing = {smoothing}")
        print(f"{'='*60}")

        # 生成平滑指数
        smoothed_index = generate_smoothed_index(raw_df, smoothing)

        # 只测试指定周期
        test_data = price_df.loc[test_start:]
        test_sentiment = smoothed_index.loc[test_start:]

        # 回测
        metrics, trades_df, portfolio_df = backtest_sentiment_strategy(
            test_data,
            test_sentiment,
            **params
        )

        results.append({
            'smoothing': smoothing,
            **metrics
        })

        # 打印结果
        print(f"\n收益率: {metrics['total_return']:.2f}%")
        print(f"夏普率: {metrics['sharpe_ratio']:.4f}")
        print(f"最大回撤: {metrics['max_drawdown']:.2f}%")
        print(f"交易次数: {metrics['num_trades']}")
        print(f"胜率: {metrics['win_rate']:.2f}%")
        print(f"最终资金: ${metrics['final_value']:.2f}")

        if len(trades_df) > 0:
            print(f"\n交易统计:")
            print(f"  平均盈亏: {trades_df['profit_pct'].mean():.2f}%")
            print(f"  最大盈利: {trades_df['profit_pct'].max():.2f}%")
            print(f"  最大亏损: {trades_df['profit_pct'].min():.2f}%")
            print(f"  平均持仓天数: {trades_df['holding_days'].mean():.1f} 天")

    # 汇总对比
    print(f"\n{'='*80}")
    print("Smoothing 参数对比汇总")
    print(f"{'='*80}\n")

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # 找出最优参数
    best_sharpe = results_df.loc[results_df['sharpe_ratio'].idxmax()]
    best_return = results_df.loc[results_df['total_return'].idxmax()]

    print(f"\n{'='*60}")
    print("最优参数分析")
    print(f"{'='*60}")
    print(f"\n夏普率最优: Smoothing = {best_sharpe['smoothing']} (夏普 {best_sharpe['sharpe_ratio']:.4f})")
    print(f"收益率最优: Smoothing = {best_return['smoothing']} (收益 {best_return['total_return']:.2f}%)")

    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'smoothing_comparison_{symbol}_{timestamp}.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ 结果已保存至: {output_file}")

    return results_df

def test_mag7_smoothing(smoothing=3):
    """测试MAG7在smoothing=3下的表现"""

    print(f"\n{'='*80}")
    print(f"MAG7 Smoothing={smoothing} 参数测试")
    print(f"测试周期: 2021-2025")
    print(f"{'='*80}\n")

    symbols = ['NVDA', 'TSLA', 'GOOGL', 'AAPL', 'MSFT', 'AMZN', 'META']

    # 新版指数参数
    params = {
        'buy_threshold': 5,
        'sell_and_threshold': 20,
        'sell_or_threshold': 40,
        'position_pct': 0.8,
        'initial_capital': 100000
    }

    results = []

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"测试 {symbol}")
        print(f"{'='*60}")

        try:
            # 加载数据
            raw_df = load_raw_index_components(symbol, "2020-01-01")
            price_df = load_price_data(symbol, "2020-01-01")

            # 对齐
            common_dates = raw_df.index.intersection(price_df.index)
            raw_df = raw_df.loc[common_dates]
            price_df = price_df.loc[common_dates]

            # 生成平滑指数
            smoothed_index = generate_smoothed_index(raw_df, smoothing)

            # 回测2021-2025
            test_data = price_df.loc["2021-01-01":]
            test_sentiment = smoothed_index.loc["2021-01-01":]

            metrics, trades_df, _ = backtest_sentiment_strategy(
                test_data,
                test_sentiment,
                **params
            )

            results.append({
                'symbol': symbol,
                **metrics
            })

            print(f"收益率: {metrics['total_return']:.2f}%")
            print(f"夏普率: {metrics['sharpe_ratio']:.4f}")
            print(f"交易次数: {metrics['num_trades']}")
            print(f"胜率: {metrics['win_rate']:.2f}%")

        except Exception as e:
            print(f"❌ {symbol} 测试失败: {e}")

    # 汇总
    results_df = pd.DataFrame(results)

    print(f"\n{'='*80}")
    print(f"MAG7 Smoothing={smoothing} 汇总结果")
    print(f"{'='*80}\n")
    print(results_df.to_string(index=False))

    print(f"\n平均收益率: {results_df['total_return'].mean():.2f}%")
    print(f"平均夏普率: {results_df['sharpe_ratio'].mean():.4f}")
    print(f"平均胜率: {results_df['win_rate'].mean():.2f}%")

    # 保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'mag7_smoothing{smoothing}_{timestamp}.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ 结果已保存至: {output_file}")

    return results_df

if __name__ == "__main__":

    # 实验1: 单股票smoothing参数对比
    print("\n" + "="*80)
    print("实验1: TSLA Smoothing 参数对比 (3, 5, 7, 10)")
    print("="*80)

    tsla_results = run_smoothing_comparison(symbol="TSLA", test_period="2021-2025")

    # 实验2: MAG7 在 smoothing=3 下的表现
    print("\n" + "="*80)
    print("实验2: MAG7 Smoothing=3 测试")
    print("="*80)

    mag7_results_s3 = test_mag7_smoothing(smoothing=3)

    # 实验3: MAG7 在 smoothing=5 下的表现（对照组）
    print("\n" + "="*80)
    print("实验3: MAG7 Smoothing=5 测试（对照组）")
    print("="*80)

    mag7_results_s5 = test_mag7_smoothing(smoothing=5)

    # 对比分析
    print("\n" + "="*80)
    print("Smoothing=3 vs Smoothing=5 对比分析")
    print("="*80)

    comparison = pd.DataFrame({
        'symbol': mag7_results_s3['symbol'],
        'return_s3': mag7_results_s3['total_return'],
        'return_s5': mag7_results_s5['total_return'],
        'sharpe_s3': mag7_results_s3['sharpe_ratio'],
        'sharpe_s5': mag7_results_s5['sharpe_ratio'],
        'trades_s3': mag7_results_s3['num_trades'],
        'trades_s5': mag7_results_s5['num_trades']
    })

    comparison['return_diff'] = comparison['return_s3'] - comparison['return_s5']
    comparison['sharpe_diff'] = comparison['sharpe_s3'] - comparison['sharpe_s5']

    print("\n", comparison.to_string(index=False))

    print(f"\n{'='*60}")
    print("汇总统计")
    print(f"{'='*60}")
    print(f"Smoothing=3 平均收益: {comparison['return_s3'].mean():.2f}%")
    print(f"Smoothing=5 平均收益: {comparison['return_s5'].mean():.2f}%")
    print(f"收益差异: {comparison['return_diff'].mean():.2f}%")

    print(f"\nSmoothing=3 平均夏普: {comparison['sharpe_s3'].mean():.4f}")
    print(f"Smoothing=5 平均夏普: {comparison['sharpe_s5'].mean():.4f}")
    print(f"夏普差异: {comparison['sharpe_diff'].mean():.4f}")

    print(f"\nSmoothing=3 平均交易次数: {comparison['trades_s3'].mean():.1f}")
    print(f"Smoothing=5 平均交易次数: {comparison['trades_s5'].mean():.1f}")

    # 保存对比结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    comparison.to_csv(f'smoothing_comparison_mag7_{timestamp}.csv', index=False)

    print(f"\n✅ 完整实验完成！")
