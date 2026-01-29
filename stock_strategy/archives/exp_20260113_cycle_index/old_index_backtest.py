"""
使用旧版指数 (fear_greed_index_backup_20260113) 进行回测
策略：
  - 买入: index < -10
  - 卖出: index > 30 OR (index > 10 AND price < MA50)
  - 时间: 2021-2025，每年独立测试
  - 初始资金: 100000，仓位80%，动态
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime

# 数据库配置
db_config = {
    "host": "localhost",
    "port": 5432,
    "database": "crypto_fear_greed_2",
    "user": "sc2025",
    "password": ""
}

# 策略参数
BUY_THRESHOLD = -10
OR_SELL_THRESHOLD = 30
AND_SELL_THRESHOLD = 10
INITIAL_CAPITAL = 100000
POSITION_PCT = 0.8
COMMISSION_RATE = 0.001
SLIPPAGE_RATE = 0.001

# MAG7 股票
SYMBOLS = ['TSLA', 'NVDA', 'META', 'AAPL', 'MSFT', 'GOOGL', 'AMZN']

# 测试年份
YEARS = [2021, 2022, 2023, 2024, 2025]


def load_old_index(symbol):
    """加载旧版指数数据"""
    conn = psycopg2.connect(**db_config)
    query = f"""
        SELECT date, smoothed_index
        FROM fear_greed_index_backup_20260113
        WHERE symbol = '{symbol}'
        ORDER BY date
    """
    df = pd.read_sql(query, conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    return df


def load_price_data(symbol):
    """加载价格数据并计算MA50"""
    conn = psycopg2.connect(**db_config)
    query = f"""
        SELECT
            TO_DATE(TO_CHAR(TO_TIMESTAMP(open_time/1000), 'YYYY-MM-DD'), 'YYYY-MM-DD') as date,
            open, high, low, close, volume
        FROM candles
        WHERE symbol = '{symbol}'
        ORDER BY open_time
    """
    df = pd.read_sql(query, conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop_duplicates(subset=['date'], keep='first')
    df = df.set_index('date')
    df['MA50'] = df['close'].rolling(window=50).mean()
    return df


def run_yearly_backtest(symbol, year, index_data, price_data):
    """运行单年回测"""
    start_date = pd.Timestamp(f'{year}-01-01')
    end_date = pd.Timestamp(f'{year}-12-31')

    # 筛选数据
    mask = (price_data.index >= start_date) & (price_data.index <= end_date)
    test_price = price_data[mask].copy()

    if len(test_price) < 10:
        return None

    # 对齐指数数据
    test_index = index_data.reindex(test_price.index)

    # 初始化
    capital = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    entry_date = None
    trades = []
    daily_values = []

    for i, date in enumerate(test_price.index):
        close = test_price.loc[date, 'close']
        ma50 = test_price.loc[date, 'MA50']
        idx_value = test_index.loc[date, 'smoothed_index'] if date in test_index.index and pd.notna(test_index.loc[date, 'smoothed_index']) else None

        # 计算当前总值
        if position > 0:
            current_value = capital + position * close
        else:
            current_value = capital

        daily_values.append({'date': date, 'value': current_value})

        if idx_value is None:
            continue

        # 卖出逻辑
        if position > 0:
            # 卖出条件: index > 30 OR (index > 10 AND price < MA50)
            or_condition = idx_value > OR_SELL_THRESHOLD
            and_condition = idx_value > AND_SELL_THRESHOLD and pd.notna(ma50) and close < ma50

            if or_condition or and_condition:
                sell_price = close * (1 - SLIPPAGE_RATE) * (1 - COMMISSION_RATE)
                proceeds = position * sell_price
                profit = proceeds - (position * entry_price)
                profit_pct = (sell_price / entry_price - 1) * 100

                trades.append({
                    'entry_date': entry_date,
                    'exit_date': date,
                    'entry_price': entry_price,
                    'exit_price': sell_price,
                    'shares': position,
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'exit_reason': 'or_threshold' if or_condition else 'and_condition'
                })

                capital = capital + proceeds
                position = 0
                entry_price = 0
                entry_date = None

        # 买入逻辑
        elif position == 0 and idx_value < BUY_THRESHOLD:
            # 动态仓位: 当前总资金 * 80%
            target_value = current_value * POSITION_PCT
            buy_price = close * (1 + SLIPPAGE_RATE) * (1 + COMMISSION_RATE)
            shares = int(target_value / buy_price)

            if shares > 0 and capital >= shares * buy_price:
                cost = shares * buy_price
                capital = capital - cost
                position = shares
                entry_price = buy_price
                entry_date = date

    # 年末强制平仓
    if position > 0:
        last_date = test_price.index[-1]
        last_close = test_price.iloc[-1]['close']
        sell_price = last_close * (1 - SLIPPAGE_RATE) * (1 - COMMISSION_RATE)
        proceeds = position * sell_price
        profit = proceeds - (position * entry_price)
        profit_pct = (sell_price / entry_price - 1) * 100

        trades.append({
            'entry_date': entry_date,
            'exit_date': last_date,
            'entry_price': entry_price,
            'exit_price': sell_price,
            'shares': position,
            'profit': profit,
            'profit_pct': profit_pct,
            'exit_reason': 'year_end'
        })

        capital = capital + proceeds
        position = 0

    # 计算指标
    if len(daily_values) == 0:
        return None

    values_df = pd.DataFrame(daily_values)
    final_value = capital
    total_return = (final_value / INITIAL_CAPITAL - 1) * 100

    # 最大回撤
    values_df['peak'] = values_df['value'].cummax()
    values_df['drawdown'] = (values_df['value'] - values_df['peak']) / values_df['peak'] * 100
    max_drawdown = values_df['drawdown'].min()

    # 夏普率 (假设无风险利率=0)
    values_df['daily_return'] = values_df['value'].pct_change()
    daily_returns = values_df['daily_return'].dropna()
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    else:
        sharpe_ratio = 0

    # 胜率
    if len(trades) > 0:
        win_trades = [t for t in trades if t['profit'] > 0]
        win_rate = len(win_trades) / len(trades) * 100
    else:
        win_rate = 0

    return {
        'symbol': symbol,
        'year': year,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'final_value': final_value,
        'trades': trades
    }


def main():
    print("=" * 80)
    print("旧版指数 (fear_greed_index_backup_20260113) 回测策略")
    print("=" * 80)
    print(f"\n策略参数:")
    print(f"  买入: 指数 < {BUY_THRESHOLD}")
    print(f"  卖出: 指数 > {OR_SELL_THRESHOLD} OR (指数 > {AND_SELL_THRESHOLD} AND 价格 < MA50)")
    print(f"  初始资金: ${INITIAL_CAPITAL:,}")
    print(f"  仓位比例: {POSITION_PCT*100}% (动态)")
    print(f"  手续费: {COMMISSION_RATE*100}%")
    print(f"  滑点: {SLIPPAGE_RATE*100}%")

    all_results = []
    all_trades = []

    for symbol in SYMBOLS:
        print(f"\n{'='*80}")
        print(f"股票: {symbol}")
        print("=" * 80)

        # 加载数据
        try:
            index_data = load_old_index(symbol)
            price_data = load_price_data(symbol)
        except Exception as e:
            print(f"  加载数据失败: {e}")
            continue

        print(f"  指数数据: {len(index_data)} 行")
        print(f"  价格数据: {len(price_data)} 行")

        symbol_results = []

        for year in YEARS:
            result = run_yearly_backtest(symbol, year, index_data, price_data)

            if result:
                symbol_results.append(result)
                all_results.append(result)

                for trade in result['trades']:
                    trade['symbol'] = symbol
                    trade['year'] = year
                    all_trades.append(trade)

                print(f"\n  {year}年:")
                print(f"    收益率: {result['total_return']:+.2f}%")
                print(f"    最大回撤: {result['max_drawdown']:.2f}%")
                print(f"    夏普率: {result['sharpe_ratio']:.2f}")
                print(f"    交易次数: {result['total_trades']}")
                print(f"    胜率: {result['win_rate']:.1f}%")

        # 汇总该股票
        if symbol_results:
            avg_return = np.mean([r['total_return'] for r in symbol_results])
            cumulative = np.prod([1 + r['total_return']/100 for r in symbol_results]) * 100 - 100
            positive_years = sum([1 for r in symbol_results if r['total_return'] > 0])

            print(f"\n  {symbol} 汇总:")
            print(f"    平均年化: {avg_return:+.2f}%")
            print(f"    5年累计: {cumulative:+.2f}%")
            print(f"    盈利年数: {positive_years}/5")

    # 输出汇总表
    print("\n" + "=" * 80)
    print("全部股票年度汇总")
    print("=" * 80)

    # 按股票汇总
    summary_data = []
    for symbol in SYMBOLS:
        symbol_results = [r for r in all_results if r['symbol'] == symbol]
        if symbol_results:
            years_dict = {r['year']: r['total_return'] for r in symbol_results}
            avg_return = np.mean([r['total_return'] for r in symbol_results])
            cumulative = np.prod([1 + r['total_return']/100 for r in symbol_results]) * 100 - 100
            positive_years = sum([1 for r in symbol_results if r['total_return'] > 0])

            row = {
                'symbol': symbol,
                '2021': years_dict.get(2021, 0),
                '2022': years_dict.get(2022, 0),
                '2023': years_dict.get(2023, 0),
                '2024': years_dict.get(2024, 0),
                '2025': years_dict.get(2025, 0),
                'avg': avg_return,
                'cumulative': cumulative,
                'positive': positive_years
            }
            summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('cumulative', ascending=False)

    print("\n股票   │  2021  │  2022  │  2023  │  2024  │  2025  │  平均  │  累计  │盈利年")
    print("─" * 85)
    for _, row in summary_df.iterrows():
        print(f"{row['symbol']:6} │ {row['2021']:+6.1f}% │ {row['2022']:+6.1f}% │ {row['2023']:+6.1f}% │ {row['2024']:+6.1f}% │ {row['2025']:+6.1f}% │ {row['avg']:+6.1f}% │ {row['cumulative']:+7.1f}% │ {int(row['positive'])}/5")

    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 保存汇总
    summary_file = f"old_index_backtest_summary_{timestamp}.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\n汇总已保存: {summary_file}")

    # 保存交易明细
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_file = f"old_index_backtest_trades_{timestamp}.csv"
        trades_df.to_csv(trades_file, index=False)
        print(f"交易明细已保存: {trades_file}")

    # 保存详细结果
    results_df = pd.DataFrame(all_results)
    results_df = results_df.drop(columns=['trades'])
    results_file = f"old_index_backtest_results_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    print(f"详细结果已保存: {results_file}")


if __name__ == "__main__":
    main()
