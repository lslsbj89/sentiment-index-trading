"""
cycle_index 统一参数回测
参数: buy<0, AND>15, OR>35
时间: 2021-2025
标的: MAG7
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import psycopg2
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# 数据库配置
db_config = {
    "host": "localhost",
    "port": 5432,
    "database": "crypto_fear_greed_2",
    "user": "sc2025",
    "password": ""
}

# 统一参数
BUY_THRESHOLD = 0
AND_SELL_THRESHOLD = 15
OR_SELL_THRESHOLD = 35

# 回测参数
INITIAL_CAPITAL = 100000
POSITION_PCT = 0.8
COMMISSION_RATE = 0.001
SLIPPAGE_RATE = 0.001

# MAG7 股票
SYMBOLS = ['NVDA', 'TSLA', 'AAPL', 'META', 'GOOGL', 'MSFT', 'AMZN']

# 测试周期
START_DATE = '2021-01-01'
END_DATE = '2025-12-31'


def load_cycle_index(symbol):
    """加载 cycle_index 数据"""
    conn = psycopg2.connect(**db_config)
    query = f"""
        SELECT date, smoothed_index
        FROM cycle_index
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


def run_backtest(symbol, index_data, price_data):
    """运行回测"""
    start_date = pd.Timestamp(START_DATE)
    end_date = pd.Timestamp(END_DATE)

    mask = (price_data.index >= start_date) & (price_data.index <= end_date)
    test_price = price_data[mask].copy()

    if len(test_price) < 10:
        return None, None, None

    test_index = index_data.reindex(test_price.index)

    # 初始化
    capital = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    entry_date = None
    entry_idx = None
    trades = []
    daily_values = []

    for date in test_price.index:
        close = test_price.loc[date, 'close']
        ma50 = test_price.loc[date, 'MA50']
        idx_value = test_index.loc[date, 'smoothed_index'] if date in test_index.index and pd.notna(test_index.loc[date, 'smoothed_index']) else None

        # 计算当前总值
        if position > 0:
            current_value = capital + position * close
        else:
            current_value = capital

        daily_values.append({
            'date': date,
            'value': current_value,
            'close': close,
            'index': idx_value,
            'position': position
        })

        if idx_value is None:
            continue

        # 卖出逻辑
        if position > 0:
            or_condition = idx_value > OR_SELL_THRESHOLD
            and_condition = idx_value > AND_SELL_THRESHOLD and pd.notna(ma50) and close < ma50

            if or_condition or and_condition:
                sell_price = close * (1 - SLIPPAGE_RATE) * (1 - COMMISSION_RATE)
                proceeds = position * sell_price
                profit = proceeds - (position * entry_price)
                profit_pct = (sell_price / entry_price - 1) * 100
                holding_days = (date - entry_date).days

                trades.append({
                    'entry_date': entry_date,
                    'entry_price': entry_price / (1 + SLIPPAGE_RATE) / (1 + COMMISSION_RATE),
                    'entry_idx': entry_idx,
                    'exit_date': date,
                    'exit_price': close,
                    'exit_idx': idx_value,
                    'shares': position,
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'holding_days': holding_days,
                    'exit_reason': 'idx>' + str(OR_SELL_THRESHOLD) if or_condition else f'idx>{AND_SELL_THRESHOLD} & <MA50'
                })

                capital = capital + proceeds
                position = 0
                entry_price = 0
                entry_date = None

        # 买入逻辑
        elif position == 0 and idx_value < BUY_THRESHOLD:
            target_value = current_value * POSITION_PCT
            buy_price = close * (1 + SLIPPAGE_RATE) * (1 + COMMISSION_RATE)
            shares = int(target_value / buy_price)

            if shares > 0 and capital >= shares * buy_price:
                cost = shares * buy_price
                capital = capital - cost
                position = shares
                entry_price = buy_price
                entry_date = date
                entry_idx = idx_value

    # 期末如有持仓，记录未平仓
    final_date = test_price.index[-1]
    final_close = test_price.iloc[-1]['close']
    final_idx = test_index.loc[final_date, 'smoothed_index'] if final_date in test_index.index else None

    if position > 0:
        sell_price = final_close * (1 - SLIPPAGE_RATE) * (1 - COMMISSION_RATE)
        proceeds = position * sell_price
        profit = proceeds - (position * entry_price)
        profit_pct = (sell_price / entry_price - 1) * 100
        holding_days = (final_date - entry_date).days

        trades.append({
            'entry_date': entry_date,
            'entry_price': entry_price / (1 + SLIPPAGE_RATE) / (1 + COMMISSION_RATE),
            'entry_idx': entry_idx,
            'exit_date': final_date,
            'exit_price': final_close,
            'exit_idx': final_idx,
            'shares': position,
            'profit': profit,
            'profit_pct': profit_pct,
            'holding_days': holding_days,
            'exit_reason': 'end_of_period'
        })

        capital = capital + proceeds
        position = 0

    # 计算指标
    values_df = pd.DataFrame(daily_values)
    final_value = capital
    total_return = (final_value / INITIAL_CAPITAL - 1) * 100

    # 最大回撤
    values_df['peak'] = values_df['value'].cummax()
    values_df['drawdown'] = (values_df['value'] - values_df['peak']) / values_df['peak'] * 100
    max_drawdown = values_df['drawdown'].min()

    # 夏普率
    values_df['daily_return'] = values_df['value'].pct_change()
    daily_returns = values_df['daily_return'].dropna()
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    else:
        sharpe_ratio = 0

    # 胜率
    if trades:
        win_trades = [t for t in trades if t['profit'] > 0]
        win_rate = len(win_trades) / len(trades) * 100
    else:
        win_rate = 0

    metrics = {
        'symbol': symbol,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'final_value': final_value
    }

    return metrics, trades, values_df


def plot_trading_chart(symbol, price_data, index_data, trades, values_df, output_dir):
    """绘制交易图表"""
    start_date = pd.Timestamp(START_DATE)
    end_date = pd.Timestamp(END_DATE)

    mask = (price_data.index >= start_date) & (price_data.index <= end_date)
    test_price = price_data[mask].copy()
    test_index = index_data.reindex(test_price.index)

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1.5, 1.5]})

    # 图1: 价格 + 交易信号
    ax1 = axes[0]
    ax1.plot(test_price.index, test_price['close'], 'b-', linewidth=1, alpha=0.8, label='Price')
    ax1.plot(test_price.index, test_price['MA50'], 'orange', linewidth=1, alpha=0.6, label='MA50')

    # 标记买入卖出点
    for trade in trades:
        # 买入点
        ax1.scatter(trade['entry_date'], trade['entry_price'],
                   color='green', marker='^', s=150, zorder=5,
                   edgecolors='darkgreen', linewidths=1.5)
        # 卖出点
        color = 'red' if trade['profit_pct'] > 0 else 'orange'
        ax1.scatter(trade['exit_date'], trade['exit_price'],
                   color=color, marker='v', s=150, zorder=5,
                   edgecolors='dark'+color if color == 'red' else 'darkorange', linewidths=1.5)

    ax1.set_title(f'{symbol} Trading Chart (2021-2025)\nStrategy: buy<{BUY_THRESHOLD}, sell>{OR_SELL_THRESHOLD} OR (>{AND_SELL_THRESHOLD} & <MA50)', fontsize=14)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    # 图2: 情绪指数
    ax2 = axes[1]
    ax2.plot(test_index.index, test_index['smoothed_index'], 'purple', linewidth=1, alpha=0.8)
    ax2.axhline(y=BUY_THRESHOLD, color='green', linestyle='--', alpha=0.5, label=f'Buy < {BUY_THRESHOLD}')
    ax2.axhline(y=AND_SELL_THRESHOLD, color='orange', linestyle='--', alpha=0.5, label=f'AND Sell > {AND_SELL_THRESHOLD}')
    ax2.axhline(y=OR_SELL_THRESHOLD, color='red', linestyle='--', alpha=0.5, label=f'OR Sell > {OR_SELL_THRESHOLD}')
    ax2.fill_between(test_index.index, test_index['smoothed_index'], BUY_THRESHOLD,
                     where=test_index['smoothed_index'] < BUY_THRESHOLD,
                     color='green', alpha=0.3)
    ax2.fill_between(test_index.index, test_index['smoothed_index'], OR_SELL_THRESHOLD,
                     where=test_index['smoothed_index'] > OR_SELL_THRESHOLD,
                     color='red', alpha=0.3)
    ax2.set_ylabel('Sentiment Index', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    # 图3: 组合价值 + 回撤
    ax3 = axes[2]
    ax3.plot(values_df['date'], values_df['value'], 'b-', linewidth=1.5, label='Portfolio Value')
    ax3.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    # 添加回撤阴影
    ax3_twin = ax3.twinx()
    ax3_twin.fill_between(values_df['date'], values_df['drawdown'], 0,
                          color='red', alpha=0.2, label='Drawdown')
    ax3_twin.set_ylabel('Drawdown (%)', fontsize=12, color='red')
    ax3_twin.tick_params(axis='y', labelcolor='red')
    ax3_twin.set_ylim(-60, 10)

    plt.tight_layout()

    # 保存图片
    output_file = os.path.join(output_dir, f'{symbol}_unified_trades.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    return output_file


def main():
    print("=" * 80)
    print("cycle_index 统一参数回测 (2021-2025)")
    print("=" * 80)
    print(f"\n策略参数:")
    print(f"  买入: 指数 < {BUY_THRESHOLD}")
    print(f"  卖出: 指数 > {OR_SELL_THRESHOLD} OR (指数 > {AND_SELL_THRESHOLD} AND 价格 < MA50)")
    print(f"  初始资金: ${INITIAL_CAPITAL:,}")
    print(f"  仓位比例: {POSITION_PCT*100}% (动态)")

    # 创建输出目录
    output_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    all_metrics = []
    all_trades = []

    for symbol in SYMBOLS:
        print(f"\n{'='*80}")
        print(f"回测: {symbol}")
        print("=" * 80)

        # 加载数据
        try:
            index_data = load_cycle_index(symbol)
            price_data = load_price_data(symbol)
        except Exception as e:
            print(f"  加载数据失败: {e}")
            continue

        # 运行回测
        metrics, trades, values_df = run_backtest(symbol, index_data, price_data)

        if metrics is None:
            print(f"  回测失败")
            continue

        all_metrics.append(metrics)

        # 打印结果
        print(f"\n  总收益: {metrics['total_return']:+.2f}%")
        print(f"  最大回撤: {metrics['max_drawdown']:.2f}%")
        print(f"  夏普率: {metrics['sharpe_ratio']:.2f}")
        print(f"  交易次数: {metrics['total_trades']}")
        print(f"  胜率: {metrics['win_rate']:.1f}%")
        print(f"  最终资金: ${metrics['final_value']:,.0f}")

        # 打印交易明细
        print(f"\n  交易明细:")
        print(f"  {'#':<3} │ {'买入日期':<12} │ {'买入价':>10} │ {'指数':>6} │ {'卖出日期':<12} │ {'卖出价':>10} │ {'指数':>6} │ {'收益':>8} │ {'天数':>5} │ {'退出原因'}")
        print(f"  {'─'*120}")

        for i, trade in enumerate(trades, 1):
            trade['symbol'] = symbol
            all_trades.append(trade)

            entry_idx = f"{trade['entry_idx']:.1f}" if trade['entry_idx'] is not None else "N/A"
            exit_idx = f"{trade['exit_idx']:.1f}" if trade['exit_idx'] is not None else "N/A"

            print(f"  {i:<3} │ {trade['entry_date'].strftime('%Y-%m-%d'):<12} │ {trade['entry_price']:>10.2f} │ {entry_idx:>6} │ {trade['exit_date'].strftime('%Y-%m-%d'):<12} │ {trade['exit_price']:>10.2f} │ {exit_idx:>6} │ {trade['profit_pct']:>+7.1f}% │ {trade['holding_days']:>5} │ {trade['exit_reason']}")

        # 绘制图表
        chart_file = plot_trading_chart(symbol, price_data, index_data, trades, values_df, output_dir)
        print(f"\n  图表已保存: {chart_file}")

    # 汇总结果
    print("\n" + "=" * 80)
    print("MAG7 汇总结果")
    print("=" * 80)

    print(f"\n{'股票':<6} │ {'收益率':>12} │ {'最大回撤':>10} │ {'夏普率':>8} │ {'交易数':>6} │ {'胜率':>8} │ {'最终资金':>12}")
    print("─" * 85)

    for m in sorted(all_metrics, key=lambda x: x['total_return'], reverse=True):
        print(f"{m['symbol']:<6} │ {m['total_return']:>+11.1f}% │ {m['max_drawdown']:>9.1f}% │ {m['sharpe_ratio']:>8.2f} │ {m['total_trades']:>6} │ {m['win_rate']:>7.1f}% │ ${m['final_value']:>11,.0f}")

    # 计算平均值
    avg_return = np.mean([m['total_return'] for m in all_metrics])
    avg_dd = np.mean([m['max_drawdown'] for m in all_metrics])
    avg_sharpe = np.mean([m['sharpe_ratio'] for m in all_metrics])
    avg_trades = np.mean([m['total_trades'] for m in all_metrics])
    avg_winrate = np.mean([m['win_rate'] for m in all_metrics])

    print("─" * 85)
    print(f"{'平均':<6} │ {avg_return:>+11.1f}% │ {avg_dd:>9.1f}% │ {avg_sharpe:>8.2f} │ {avg_trades:>6.1f} │ {avg_winrate:>7.1f}% │")

    # 保存结果
    # 保存汇总
    summary_df = pd.DataFrame(all_metrics)
    summary_file = os.path.join(output_dir, f'unified_backtest_summary_{timestamp}.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"\n汇总已保存: {summary_file}")

    # 保存交易明细
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_file = os.path.join(output_dir, f'unified_backtest_trades_{timestamp}.csv')
        trades_df.to_csv(trades_file, index=False)
        print(f"交易明细已保存: {trades_file}")

    print("\n" + "=" * 80)
    print("回测完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
