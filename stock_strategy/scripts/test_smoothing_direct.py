"""
Smoothing 参数对比实验 (直接使用数据库中的数据)

利用已有的 fear_greed_index_s3 (smoothing=3) 和 fear_greed_index (smoothing=5)
"""

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

def load_smoothed_index(symbol, smoothing=5, start_date="2016-01-01"):
    """加载指定smoothing的平滑指数"""
    conn = psycopg2.connect(**db_config)

    table_name = "fear_greed_index_s3" if smoothing == 3 else "fear_greed_index"

    query = f"""
        SELECT date, smoothed_index
        FROM {table_name}
        WHERE symbol = '{symbol}' AND date >= '{start_date}'
        ORDER BY date
    """

    df = pd.read_sql(query, conn)
    conn.close()

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    return df

def load_price_data(symbol, start_date="2016-01-01"):
    """加载价格数据"""
    conn = psycopg2.connect(**db_config)

    query = f"""
        SELECT date, open, high, low, close, volume
        FROM candles
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

def test_mag7_smoothing_comparison():
    """测试MAG7在smoothing=3和5下的表现对比"""

    print(f"\n{'='*80}")
    print(f"MAG7 Smoothing 参数对比实验 (smoothing=3 vs smoothing=5)")
    print(f"测试周期: 2021-2025")
    print(f"{'='*80}\n")

    symbols = ['NVDA', 'TSLA', 'GOOGL', 'AAPL', 'MSFT', 'AMZN', 'META']

    # 策略参数
    params = {
        'buy_threshold': 5,
        'sell_and_threshold': 20,
        'sell_or_threshold': 40,
        'position_pct': 0.8,
        'initial_capital': 100000
    }

    results_s3 = []
    results_s5 = []

    for symbol in symbols:
        print(f"\n{'='*70}")
        print(f"测试 {symbol}")
        print(f"{'='*70}")

        try:
            # 加载价格数据
            price_df = load_price_data(symbol, "2020-01-01")

            # 测试 smoothing=3
            print(f"\n  Smoothing=3:")
            sentiment_s3 = load_smoothed_index(symbol, smoothing=3, start_date="2020-01-01")

            # 对齐数据
            common_dates = price_df.index.intersection(sentiment_s3.index)
            test_price = price_df.loc[common_dates].loc["2021-01-01":]
            test_sentiment = sentiment_s3.loc[common_dates].loc["2021-01-01":]

            metrics_s3, trades_s3, _ = backtest_sentiment_strategy(
                test_price,
                test_sentiment['smoothed_index'],
                **params
            )

            results_s3.append({
                'symbol': symbol,
                **metrics_s3
            })

            print(f"    收益率: {metrics_s3['total_return']:.2f}%")
            print(f"    夏普率: {metrics_s3['sharpe_ratio']:.4f}")
            print(f"    最大回撤: {metrics_s3['max_drawdown']:.2f}%")
            print(f"    交易次数: {metrics_s3['num_trades']}")
            print(f"    胜率: {metrics_s3['win_rate']:.2f}%")

            # 测试 smoothing=5
            print(f"\n  Smoothing=5:")
            sentiment_s5 = load_smoothed_index(symbol, smoothing=5, start_date="2020-01-01")

            common_dates = price_df.index.intersection(sentiment_s5.index)
            test_price = price_df.loc[common_dates].loc["2021-01-01":]
            test_sentiment = test_sentiment = sentiment_s5.loc[common_dates].loc["2021-01-01":]

            metrics_s5, trades_s5, _ = backtest_sentiment_strategy(
                test_price,
                test_sentiment['smoothed_index'],
                **params
            )

            results_s5.append({
                'symbol': symbol,
                **metrics_s5
            })

            print(f"    收益率: {metrics_s5['total_return']:.2f}%")
            print(f"    夏普率: {metrics_s5['sharpe_ratio']:.4f}")
            print(f"    最大回撤: {metrics_s5['max_drawdown']:.2f}%")
            print(f"    交易次数: {metrics_s5['num_trades']}")
            print(f"    胜率: {metrics_s5['win_rate']:.2f}%")

            # 对比
            return_diff = metrics_s3['total_return'] - metrics_s5['total_return']
            sharpe_diff = metrics_s3['sharpe_ratio'] - metrics_s5['sharpe_ratio']

            print(f"\n  📊 对比:")
            print(f"    收益差异: {return_diff:+.2f}% ({'✅ s3更优' if return_diff > 0 else '⚠️ s5更优'})")
            print(f"    夏普差异: {sharpe_diff:+.4f} ({'✅ s3更优' if sharpe_diff > 0 else '⚠️ s5更优'})")
            print(f"    交易次数差异: {metrics_s3['num_trades'] - metrics_s5['num_trades']:+d}")

        except Exception as e:
            print(f"❌ {symbol} 测试失败: {e}")

    # 汇总对比
    print(f"\n{'='*80}")
    print("汇总对比分析")
    print(f"{'='*80}\n")

    df_s3 = pd.DataFrame(results_s3)
    df_s5 = pd.DataFrame(results_s5)

    comparison = pd.DataFrame({
        'symbol': df_s3['symbol'],
        'return_s3': df_s3['total_return'],
        'return_s5': df_s5['total_return'],
        'return_diff': df_s3['total_return'] - df_s5['total_return'],
        'sharpe_s3': df_s3['sharpe_ratio'],
        'sharpe_s5': df_s5['sharpe_ratio'],
        'sharpe_diff': df_s3['sharpe_ratio'] - df_s5['sharpe_ratio'],
        'trades_s3': df_s3['num_trades'],
        'trades_s5': df_s5['num_trades'],
        'win_rate_s3': df_s3['win_rate'],
        'win_rate_s5': df_s5['win_rate']
    })

    print(comparison.to_string(index=False))

    # 统计
    print(f"\n{'='*70}")
    print("统计摘要")
    print(f"{'='*70}")

    print(f"\n📈 平均收益率:")
    print(f"  Smoothing=3: {comparison['return_s3'].mean():.2f}%")
    print(f"  Smoothing=5: {comparison['return_s5'].mean():.2f}%")
    print(f"  差异: {comparison['return_diff'].mean():+.2f}%")

    print(f"\n📊 平均夏普率:")
    print(f"  Smoothing=3: {comparison['sharpe_s3'].mean():.4f}")
    print(f"  Smoothing=5: {comparison['sharpe_s5'].mean():.4f}")
    print(f"  差异: {comparison['sharpe_diff'].mean():+.4f}")

    print(f"\n🔄 平均交易次数:")
    print(f"  Smoothing=3: {comparison['trades_s3'].mean():.1f}")
    print(f"  Smoothing=5: {comparison['trades_s5'].mean():.1f}")
    print(f"  差异: {comparison['trades_s3'].mean() - comparison['trades_s5'].mean():+.1f}")

    print(f"\n🎯 平均胜率:")
    print(f"  Smoothing=3: {comparison['win_rate_s3'].mean():.2f}%")
    print(f"  Smoothing=5: {comparison['win_rate_s5'].mean():.2f}%")

    # 胜负统计
    s3_wins = (comparison['return_diff'] > 0).sum()
    s5_wins = (comparison['return_diff'] < 0).sum()

    print(f"\n🏆 胜负比 (按收益):")
    print(f"  Smoothing=3 胜出: {s3_wins}/7 股票")
    print(f"  Smoothing=5 胜出: {s5_wins}/7 股票")

    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    df_s3.to_csv(f'mag7_smoothing3_{timestamp}.csv', index=False)
    df_s5.to_csv(f'mag7_smoothing5_{timestamp}.csv', index=False)
    comparison.to_csv(f'smoothing_comparison_mag7_{timestamp}.csv', index=False)

    print(f"\n✅ 结果已保存:")
    print(f"  - mag7_smoothing3_{timestamp}.csv")
    print(f"  - mag7_smoothing5_{timestamp}.csv")
    print(f"  - smoothing_comparison_mag7_{timestamp}.csv")

    return comparison

if __name__ == "__main__":

    print("\n" + "="*80)
    print("Smoothing 参数对比实验")
    print("利用数据库现有数据: fear_greed_index_s3 vs fear_greed_index")
    print("="*80)

    results = test_mag7_smoothing_comparison()

    print(f"\n{'='*80}")
    print("✅ 实验完成！")
    print(f"{'='*80}\n")
