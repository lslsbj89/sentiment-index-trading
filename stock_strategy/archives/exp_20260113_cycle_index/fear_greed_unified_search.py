#!/usr/bin/env python3
"""
fear_greed_index 统一参数搜索
目标：找到适用于所有MAG7股票的统一参数
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, '/Users/sc2025/Desktop/Claude/Quantrade/src')

import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader

# 数据库配置
db_config = {
    'host': 'localhost',
    'database': 'crypto_fear_greed_2',
    'user': 'sc2025',
    'password': '',
    'port': 5432
}

# MAG7股票
SYMBOLS = ['NVDA', 'TSLA', 'AAPL', 'META', 'GOOGL', 'MSFT', 'AMZN']

# 回测参数
INITIAL_CAPITAL = 100000
POSITION_PCT = 0.8
COMMISSION = 0.001
SLIPPAGE = 0.001


def load_fear_greed_index(symbol):
    """加载 fear_greed_index 数据 (4因子版本，无黄金)"""
    conn = psycopg2.connect(**db_config)
    query = f"""
        SELECT date, smoothed_index as fear_greed_index
        FROM fear_greed_index
        WHERE symbol = '{symbol}'
        AND date >= '2021-01-01'
        AND date <= '2025-12-31'
        ORDER BY date
    """
    df = pd.read_sql(query, conn, parse_dates=['date'])
    conn.close()
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df.set_index('date', inplace=True)
    return df


def run_backtest(prices, index_data, buy_threshold, and_threshold, or_threshold):
    """运行回测"""
    # 合并数据
    df = prices.copy()
    df['idx'] = index_data['fear_greed_index']
    df['MA50'] = df['Close'].rolling(50).mean()
    df = df.dropna()

    if len(df) < 100:
        return None

    # 初始化
    cash = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    trades = []
    portfolio_values = [INITIAL_CAPITAL]

    for i in range(1, len(df)):
        current_price = df['Close'].iloc[i]
        current_idx = df['idx'].iloc[i]
        current_ma50 = df['MA50'].iloc[i]
        current_date = df.index[i]

        # 买入条件
        if position == 0 and current_idx < buy_threshold:
            # 动态仓位
            available = cash * POSITION_PCT
            shares = int(available / (current_price * (1 + COMMISSION + SLIPPAGE)))
            if shares > 0:
                cost = shares * current_price * (1 + COMMISSION + SLIPPAGE)
                cash -= cost
                position = shares
                entry_price = current_price
                entry_date = current_date
                entry_idx = current_idx

        # 卖出条件
        elif position > 0:
            sell_signal = False
            exit_reason = ''

            # OR条件：指数 > or_threshold
            if current_idx > or_threshold:
                sell_signal = True
                exit_reason = f'idx>{or_threshold}'
            # AND条件：指数 > and_threshold 且 价格 < MA50
            elif current_idx > and_threshold and current_price < current_ma50:
                sell_signal = True
                exit_reason = f'idx>{and_threshold} & <MA50'

            if sell_signal:
                revenue = position * current_price * (1 - COMMISSION - SLIPPAGE)
                profit = revenue - (position * entry_price * (1 + COMMISSION + SLIPPAGE))
                profit_pct = (current_price - entry_price) / entry_price * 100

                trades.append({
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'entry_idx': entry_idx,
                    'exit_date': current_date,
                    'exit_price': current_price,
                    'exit_idx': current_idx,
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'exit_reason': exit_reason
                })

                cash += revenue
                position = 0

        # 记录组合价值
        portfolio_value = cash + position * current_price
        portfolio_values.append(portfolio_value)

    # 期末强制平仓
    if position > 0:
        final_price = df['Close'].iloc[-1]
        revenue = position * final_price * (1 - COMMISSION - SLIPPAGE)
        profit = revenue - (position * entry_price * (1 + COMMISSION + SLIPPAGE))
        profit_pct = (final_price - entry_price) / entry_price * 100

        trades.append({
            'entry_date': entry_date,
            'entry_price': entry_price,
            'entry_idx': entry_idx,
            'exit_date': df.index[-1],
            'exit_price': final_price,
            'exit_idx': df['idx'].iloc[-1],
            'profit': profit,
            'profit_pct': profit_pct,
            'exit_reason': 'end_of_period'
        })

        cash += revenue
        portfolio_values[-1] = cash

    # 计算指标
    final_value = portfolio_values[-1]
    total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    # 最大回撤
    portfolio_series = pd.Series(portfolio_values)
    rolling_max = portfolio_series.cummax()
    drawdown = (portfolio_series - rolling_max) / rolling_max * 100
    max_drawdown = drawdown.min()

    # 夏普率
    returns = portfolio_series.pct_change().dropna()
    if len(returns) > 0 and returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
    else:
        sharpe = 0

    # 胜率
    if len(trades) > 0:
        wins = sum(1 for t in trades if t['profit'] > 0)
        win_rate = wins / len(trades) * 100
    else:
        win_rate = 0

    return {
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'final_value': final_value
    }


def main():
    print("=" * 80)
    print("fear_greed_index 统一参数搜索")
    print("=" * 80)

    # 使用 DataLoader 加载价格数据
    loader = DataLoader(db_config)

    # 加载数据
    print("\n加载数据...")
    all_data = {}
    for symbol in SYMBOLS:
        prices = loader.load_ohlcv(symbol, start_date='2021-01-01', end_date='2025-12-31')
        index_data = load_fear_greed_index(symbol)
        if len(prices) > 0 and len(index_data) > 0:
            all_data[symbol] = {'prices': prices, 'index': index_data}
            print(f"  {symbol}: {len(prices)} 天价格, {len(index_data)} 天指数")

    loader.close()

    # 统一参数候选 (基于分股票最优参数分布)
    buy_candidates = [-10, -5, 0, 5]
    and_candidates = [10, 15, 20, 25]
    or_candidates = [25, 30, 35, 40]

    print(f"\n参数搜索范围:")
    print(f"  买入阈值: {buy_candidates}")
    print(f"  AND卖出: {and_candidates}")
    print(f"  OR卖出: {or_candidates}")

    # 网格搜索
    results = []
    total_combinations = len(buy_candidates) * len(and_candidates) * len(or_candidates)

    print(f"\n测试 {total_combinations} 种参数组合...")

    for buy_th in buy_candidates:
        for and_th in and_candidates:
            for or_th in or_candidates:
                if or_th <= and_th:  # OR阈值应大于AND阈值
                    continue

                # 对所有股票测试
                symbol_results = {}
                valid_count = 0

                for symbol, data in all_data.items():
                    result = run_backtest(
                        data['prices'], data['index'],
                        buy_th, and_th, or_th
                    )
                    if result:
                        symbol_results[symbol] = result
                        valid_count += 1

                if valid_count >= 5:  # 至少5只股票有结果
                    avg_return = np.mean([r['total_return'] for r in symbol_results.values()])
                    avg_sharpe = np.mean([r['sharpe_ratio'] for r in symbol_results.values()])
                    avg_drawdown = np.mean([r['max_drawdown'] for r in symbol_results.values()])
                    avg_win_rate = np.mean([r['win_rate'] for r in symbol_results.values()])

                    results.append({
                        'buy_threshold': buy_th,
                        'and_threshold': and_th,
                        'or_threshold': or_th,
                        'avg_return': avg_return,
                        'avg_sharpe': avg_sharpe,
                        'avg_drawdown': avg_drawdown,
                        'avg_win_rate': avg_win_rate,
                        'valid_symbols': valid_count,
                        'details': symbol_results
                    })

    # 转为DataFrame并排序
    df_results = pd.DataFrame(results)

    # 按综合指标排序 (收益率权重0.4, 夏普率权重0.4, 回撤权重0.2)
    df_results['score'] = (
        df_results['avg_return'] / df_results['avg_return'].max() * 0.4 +
        df_results['avg_sharpe'] / df_results['avg_sharpe'].max() * 0.4 +
        (100 + df_results['avg_drawdown']) / 100 * 0.2  # 回撤越小越好
    )

    df_results = df_results.sort_values('score', ascending=False)

    print("\n" + "=" * 80)
    print("Top 10 统一参数组合:")
    print("=" * 80)
    print(f"{'排名':<4} {'买入':<6} {'AND':<6} {'OR':<6} {'收益率':<10} {'夏普率':<8} {'回撤':<10} {'胜率':<8}")
    print("-" * 70)

    for i, row in df_results.head(10).iterrows():
        print(f"{df_results.index.get_loc(i)+1:<4} "
              f"<{row['buy_threshold']:<4} "
              f">{row['and_threshold']:<4} "
              f">{row['or_threshold']:<4} "
              f"{row['avg_return']:>+7.1f}%  "
              f"{row['avg_sharpe']:>6.2f}  "
              f"{row['avg_drawdown']:>7.1f}%  "
              f"{row['avg_win_rate']:>5.1f}%")

    # 最优参数详情
    best = df_results.iloc[0]
    print("\n" + "=" * 80)
    print("最优统一参数:")
    print("=" * 80)
    print(f"  买入条件: idx < {best['buy_threshold']}")
    print(f"  AND卖出: idx > {best['and_threshold']} AND price < MA50")
    print(f"  OR卖出: idx > {best['or_threshold']}")

    print(f"\n平均指标:")
    print(f"  收益率: {best['avg_return']:+.1f}%")
    print(f"  夏普率: {best['avg_sharpe']:.2f}")
    print(f"  最大回撤: {best['avg_drawdown']:.1f}%")
    print(f"  胜率: {best['avg_win_rate']:.1f}%")

    print(f"\n各股票表现:")
    print(f"{'股票':<8} {'收益率':<12} {'夏普率':<10} {'回撤':<12} {'交易数':<8} {'胜率':<8}")
    print("-" * 60)

    for symbol in SYMBOLS:
        if symbol in best['details']:
            r = best['details'][symbol]
            print(f"{symbol:<8} {r['total_return']:>+8.1f}%  "
                  f"{r['sharpe_ratio']:>8.2f}  "
                  f"{r['max_drawdown']:>8.1f}%  "
                  f"{r['num_trades']:>6}  "
                  f"{r['win_rate']:>5.1f}%")

    # 与分股票最优参数对比
    print("\n" + "=" * 80)
    print("统一参数 vs 分股票最优参数对比:")
    print("=" * 80)

    # 分股票最优参数 (来自步骤2)
    per_stock_optimal = {
        'TSLA': {'buy': -10, 'and': 20, 'or': 35, 'return': 724.6, 'sharpe': 1.42, 'drawdown': -60.8},
        'NVDA': {'buy': 0, 'and': 25, 'or': 40, 'return': 446.0, 'sharpe': 1.24, 'drawdown': -50.2},
        'GOOGL': {'buy': 5, 'and': 15, 'or': 35, 'return': 223.1, 'sharpe': 1.22, 'drawdown': -35.0},
        'AAPL': {'buy': 0, 'and': 15, 'or': 30, 'return': 151.5, 'sharpe': 1.24, 'drawdown': -24.0},
        'MSFT': {'buy': 5, 'and': 15, 'or': 30, 'return': 85.0, 'sharpe': 0.83, 'drawdown': -26.2},
        'AMZN': {'buy': -10, 'and': 20, 'or': 30, 'return': 68.2, 'sharpe': 0.66, 'drawdown': -42.6},
        'META': {'buy': 0, 'and': 15, 'or': 40, 'return': 45.8, 'sharpe': 0.76, 'drawdown': -58.8}
    }

    per_stock_avg_return = np.mean([v['return'] for v in per_stock_optimal.values()])
    per_stock_avg_sharpe = np.mean([v['sharpe'] for v in per_stock_optimal.values()])
    per_stock_avg_drawdown = np.mean([v['drawdown'] for v in per_stock_optimal.values()])

    print(f"\n{'指标':<15} {'分股票最优':<15} {'统一参数':<15} {'差异':<15}")
    print("-" * 60)
    print(f"{'平均收益率':<15} {per_stock_avg_return:>+10.1f}%  {best['avg_return']:>+10.1f}%  "
          f"{best['avg_return'] - per_stock_avg_return:>+10.1f}%")
    print(f"{'平均夏普率':<15} {per_stock_avg_sharpe:>10.2f}   {best['avg_sharpe']:>10.2f}   "
          f"{best['avg_sharpe'] - per_stock_avg_sharpe:>+10.2f}")
    print(f"{'平均回撤':<15} {per_stock_avg_drawdown:>10.1f}%  {best['avg_drawdown']:>10.1f}%  "
          f"{best['avg_drawdown'] - per_stock_avg_drawdown:>+10.1f}%")

    # 计算损失比例
    return_loss = (best['avg_return'] - per_stock_avg_return) / per_stock_avg_return * 100
    sharpe_loss = (best['avg_sharpe'] - per_stock_avg_sharpe) / per_stock_avg_sharpe * 100

    print(f"\n收益损失: {return_loss:.1f}%")
    print(f"夏普损失: {sharpe_loss:.1f}%")
    print(f"参数减少: 从21个 → 3个 (减少 {(1-3/21)*100:.0f}%)")

    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'fear_greed_unified_params_{timestamp}.csv'
    df_results[['buy_threshold', 'and_threshold', 'or_threshold',
                'avg_return', 'avg_sharpe', 'avg_drawdown', 'avg_win_rate']].to_csv(output_file, index=False)
    print(f"\n结果已保存: {output_file}")

    return best['buy_threshold'], best['and_threshold'], best['or_threshold']


if __name__ == '__main__':
    main()
