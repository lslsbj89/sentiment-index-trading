"""
Walk-Forward 详细分析脚本
- 显示训练/测试阈值参数
- 记录测试期详细交易过程
- 可视化交易过程
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from data_loader import DataLoader

# Configure fonts
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 配置
# ============================================================
SYMBOLS = ["TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"]

db_config = {
    "host": "localhost",
    "port": 5432,
    "database": "crypto_fear_greed_2",
    "user": "sc2025",
    "password": ""
}

# 网格搜索参数
BUY_THRESHOLDS = [-30, -25, -20, -15, -10, -5, 0, 5]
AND_SELL_THRESHOLDS = [5, 10, 15, 20, 25]
OR_THRESHOLDS = [25, 30, 35, 40, 45, 50, 55, 60, 65]

# 回测参数
INITIAL_CAPITAL = 100000
COMMISSION_RATE = 0.001
SLIPPAGE_RATE = 0.001
POSITION_PCT = 0.8

# 阈值放宽系数
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


def load_fear_greed_index(symbol):
    conn = psycopg2.connect(**db_config)
    query = """
        SELECT date, smoothed_index
        FROM fear_greed_index
        WHERE symbol = %s
        ORDER BY date
    """
    df = pd.read_sql(query, conn, params=(symbol,))
    conn.close()
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.set_index('date')
    return df


def run_backtest_detailed(price_data, sentiment_data, buy_threshold, and_sell_threshold,
                          or_threshold, initial_position=0, initial_entry_price=0, initial_cash=None):
    """
    运行回测，返回详细交易记录

    Returns:
        final_value, trades, end_position, end_entry_price, df, end_cash
    """
    df = price_data.copy()
    df['sentiment'] = sentiment_data['smoothed_index'].reindex(df.index)
    df['MA50'] = df['Close'].rolling(50).mean()
    df = df.dropna()

    if len(df) < 10:
        return INITIAL_CAPITAL, [], 0, 0, df, INITIAL_CAPITAL

    # 使用传入的initial_cash，如果没有则使用INITIAL_CAPITAL
    if initial_cash is not None:
        cash = initial_cash
    else:
        cash = INITIAL_CAPITAL

    position = initial_position
    entry_price = initial_entry_price
    entry_date = None
    trades = []

    if position > 0:
        entry_date = df.index[0]

    for i in range(len(df)):
        current_date = df.index[i]
        current_price = df['Close'].iloc[i]
        current_sentiment = df['sentiment'].iloc[i]
        current_ma50 = df['MA50'].iloc[i]

        # 买入
        if position == 0 and current_sentiment < buy_threshold:
            available = cash * POSITION_PCT
            buy_price = current_price * (1 + SLIPPAGE_RATE) * (1 + COMMISSION_RATE)
            shares = int(available / buy_price)

            if shares > 0:
                cost = shares * buy_price
                cash -= cost
                position = shares
                entry_price = buy_price
                entry_date = current_date

                trades.append({
                    'type': 'BUY',
                    'date': current_date,
                    'price': current_price,
                    'shares': shares,
                    'sentiment': current_sentiment,
                    'ma50': current_ma50,
                    'reason': f'sentiment {current_sentiment:.1f} < {buy_threshold}'
                })

        # 卖出
        elif position > 0:
            sell_signal = False
            exit_reason = None

            if current_sentiment > or_threshold:
                sell_signal = True
                exit_reason = f'OR: sentiment {current_sentiment:.1f} > {or_threshold}'
            elif current_sentiment > and_sell_threshold and current_price < current_ma50:
                sell_signal = True
                exit_reason = f'AND: sentiment {current_sentiment:.1f} > {and_sell_threshold} & price < MA50'

            if sell_signal:
                sell_price = current_price * (1 - SLIPPAGE_RATE) * (1 - COMMISSION_RATE)
                revenue = position * sell_price
                cash += revenue

                profit_pct = (sell_price - entry_price) / entry_price * 100

                trades.append({
                    'type': 'SELL',
                    'date': current_date,
                    'price': current_price,
                    'shares': position,
                    'sentiment': current_sentiment,
                    'ma50': current_ma50,
                    'reason': exit_reason,
                    'profit_pct': profit_pct
                })

                position = 0
                entry_price = 0

    final_value = cash + position * df['Close'].iloc[-1]
    return final_value, trades, position, entry_price, df, cash


def grid_search_train(train_price, train_sentiment):
    """训练期网格搜索，返回最优参数"""
    results = []

    for buy_t, and_t, or_t in product(BUY_THRESHOLDS, AND_SELL_THRESHOLDS, OR_THRESHOLDS):
        try:
            train_final, train_trades, end_position, end_entry_price, _, end_cash = run_backtest_detailed(
                train_price, train_sentiment, buy_t, and_t, or_t,
                initial_position=0, initial_entry_price=0
            )

            train_return = (train_final / INITIAL_CAPITAL - 1) * 100

            results.append({
                'buy_threshold': buy_t,
                'and_sell_threshold': and_t,
                'or_threshold': or_t,
                'train_return': train_return,
                'train_trades': len(train_trades),
                'end_position': end_position,
                'end_entry_price': end_entry_price,
                'end_cash': end_cash  # 添加训练期结束时的现金
            })

        except Exception as e:
            pass

    df = pd.DataFrame(results)
    best = df.sort_values('train_return', ascending=False).iloc[0]
    return best, df


def visualize_test_period(symbol, window_name, test_df, trades, params, test_params, test_return, output_dir):
    """可视化测试期交易过程"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # 子图1: 价格 + 交易点
    ax1 = axes[0]
    ax1.plot(test_df.index, test_df['Close'], 'b-', linewidth=1, label='Price')
    ax1.plot(test_df.index, test_df['MA50'], 'orange', linewidth=1, alpha=0.7, label='MA50')

    # 标记交易
    for trade in trades:
        if trade['type'] == 'BUY':
            ax1.scatter(trade['date'], trade['price'], color='green', marker='^', s=150, zorder=5, label='Buy' if trade == trades[0] else '')
            ax1.annotate(f"Buy\n${trade['price']:.1f}", (trade['date'], trade['price']),
                        textcoords="offset points", xytext=(0, 15), ha='center', fontsize=8, color='green')
        else:
            color = 'red' if trade.get('profit_pct', 0) > 0 else 'darkred'
            ax1.scatter(trade['date'], trade['price'], color=color, marker='v', s=150, zorder=5)
            profit_str = f"+{trade.get('profit_pct', 0):.1f}%" if trade.get('profit_pct', 0) > 0 else f"{trade.get('profit_pct', 0):.1f}%"
            ax1.annotate(f"Sell\n${trade['price']:.1f}\n{profit_str}", (trade['date'], trade['price']),
                        textcoords="offset points", xytext=(0, -25), ha='center', fontsize=8, color=color)

    ax1.set_ylabel('Price ($)')
    ax1.set_title(f'{symbol} - {window_name} Test Period Trading')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 子图2: 情绪指数 + 阈值线
    ax2 = axes[1]
    ax2.plot(test_df.index, test_df['sentiment'], 'purple', linewidth=1, label='Sentiment Index')
    ax2.axhline(y=test_params['buy'], color='green', linestyle='--', linewidth=1.5, label=f"Buy < {test_params['buy']}")
    ax2.axhline(y=test_params['and_sell'], color='orange', linestyle='--', linewidth=1.5, label=f"AND > {test_params['and_sell']}")
    ax2.axhline(y=test_params['or_sell'], color='red', linestyle='--', linewidth=1.5, label=f"OR > {test_params['or_sell']}")
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    # 标记交易点的情绪值
    for trade in trades:
        color = 'green' if trade['type'] == 'BUY' else 'red'
        ax2.scatter(trade['date'], trade['sentiment'], color=color, marker='o', s=100, zorder=5)

    ax2.set_ylabel('Sentiment Index')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 子图3: 参数配置和统计信息
    ax3 = axes[2]
    ax3.axis('off')

    info_text = f"""
    ===========================================================================
    PARAMETERS
    ===========================================================================
    Train Params: Buy < {params['buy']}, AND Sell > {params['and_sell']}, OR Sell > {params['or_sell']}
    Test Params:  Buy < {test_params['buy']:.0f}, AND Sell > {test_params['and_sell']}, OR Sell > {test_params['or_sell']}
    Relaxation:   {THRESHOLD_RELAX_FACTOR} (Train {params['buy']} x {THRESHOLD_RELAX_FACTOR} = Test {test_params['buy']:.0f})

    ===========================================================================
    TEST PERIOD STATS
    ===========================================================================
    Test Return:    {test_return:+.1f}%
    Trade Count:    {len(trades)}
    Sentiment Range: [{test_df['sentiment'].min():.1f}, {test_df['sentiment'].max():.1f}]
    """

    if trades:
        info_text += "\n    ===========================================================================\n"
        info_text += "    TRADE LOG\n"
        info_text += "    ===========================================================================\n"
        for i, t in enumerate(trades):
            date_str = t['date'].strftime('%Y-%m-%d')
            if t['type'] == 'BUY':
                info_text += f"    {i+1}. {date_str} BUY  @ ${t['price']:.2f} | Sentiment: {t['sentiment']:.1f}\n"
            else:
                info_text += f"    {i+1}. {date_str} SELL @ ${t['price']:.2f} | Sentiment: {t['sentiment']:.1f} | Profit: {t.get('profit_pct', 0):+.1f}%\n"

    ax3.text(0.02, 0.98, info_text, transform=ax3.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    filename = os.path.join(output_dir, f'{symbol}_{window_name}_analysis.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    return filename


def analyze_symbol(symbol, output_dir):
    """分析单个股票"""
    print(f"\n{'='*80}")
    print(f"  {symbol} 详细分析")
    print(f"{'='*80}")

    # 加载数据
    loader = DataLoader(db_config)
    price_data = loader.load_ohlcv(symbol, start_date="2011-01-01")
    loader.close()
    sentiment_data = load_fear_greed_index(symbol)

    if len(price_data) < 500:
        print(f"  数据不足，跳过")
        return None

    all_results = []

    for window in WINDOWS:
        print(f"\n  {window['name']}: Train {window['train'][0][:4]}-{window['train'][1][:4]} → Test {window['test'][0][:4]}")

        train_start = pd.Timestamp(window['train'][0], tz='UTC')
        train_end = pd.Timestamp(window['train'][1], tz='UTC')
        test_start = pd.Timestamp(window['test'][0], tz='UTC')
        test_end = pd.Timestamp(window['test'][1], tz='UTC')

        train_price = price_data[(price_data.index >= train_start) & (price_data.index <= train_end)]
        test_price = price_data[(price_data.index >= test_start) & (price_data.index <= test_end)]
        train_sentiment = sentiment_data[(sentiment_data.index >= train_start) & (sentiment_data.index <= train_end)]
        test_sentiment = sentiment_data[(sentiment_data.index >= test_start) & (sentiment_data.index <= test_end)]

        if len(train_price) < 100 or len(test_price) < 50:
            print(f"    数据不足，跳过")
            continue

        # 训练期网格搜索
        best_params, grid_results = grid_search_train(train_price, train_sentiment)

        train_params = {
            'buy': int(best_params['buy_threshold']),
            'and_sell': int(best_params['and_sell_threshold']),
            'or_sell': int(best_params['or_threshold'])
        }

        test_params = {
            'buy': train_params['buy'] * THRESHOLD_RELAX_FACTOR,
            'and_sell': train_params['and_sell'],
            'or_sell': train_params['or_sell']
        }

        print(f"    训练期最优参数: Buy<{train_params['buy']}, AND>{train_params['and_sell']}, OR>{train_params['or_sell']}")
        print(f"    训练期收益: {best_params['train_return']:+.1f}% ({int(best_params['train_trades'])}次交易)")
        print(f"    测试期参数: Buy<{test_params['buy']:.0f}, AND>{test_params['and_sell']}, OR>{test_params['or_sell']}")

        # 计算测试期初始价值
        end_position = int(best_params['end_position'])
        end_entry_price = best_params['end_entry_price']
        end_cash = best_params['end_cash']  # 训练期结束时的实际现金

        if end_position > 0:
            test_first_price = test_price['Close'].iloc[0]
            # 正确计算：使用训练期结束时的实际现金 + 持仓按测试期首日价格估值
            test_start_value = end_cash + end_position * test_first_price
            print(f"    持仓延续: {end_position}股 @ ${end_entry_price:.2f}, 现金: ${end_cash:,.0f}")
        else:
            test_start_value = end_cash  # 空仓时，使用训练期结束时的现金
            print(f"    持仓延续: 空仓, 现金: ${end_cash:,.0f}")

        # 测试期回测 - 传入正确的初始现金
        test_final, test_trades, _, _, test_df, _ = run_backtest_detailed(
            test_price, test_sentiment,
            test_params['buy'], test_params['and_sell'], test_params['or_sell'],
            initial_position=end_position, initial_entry_price=end_entry_price,
            initial_cash=end_cash  # 使用训练期结束时的实际现金
        )

        test_return = (test_final / test_start_value - 1) * 100
        print(f"    测试期收益: {test_return:+.1f}% ({len(test_trades)}次交易)")

        # 打印交易详情
        if test_trades:
            print(f"    交易记录:")
            for t in test_trades:
                date_str = t['date'].strftime('%Y-%m-%d')
                if t['type'] == 'BUY':
                    print(f"      {date_str} BUY  @ ${t['price']:.2f} | Sentiment: {t['sentiment']:.1f}")
                else:
                    print(f"      {date_str} SELL @ ${t['price']:.2f} | Sentiment: {t['sentiment']:.1f} | {t.get('profit_pct', 0):+.1f}%")

        # 可视化
        fig_path = visualize_test_period(
            symbol, window['name'], test_df, test_trades,
            train_params, test_params, test_return, output_dir
        )
        print(f"    图表已保存: {os.path.basename(fig_path)}")

        all_results.append({
            'symbol': symbol,
            'window': window['name'],
            'test_year': window['test'][0][:4],
            'train_buy': train_params['buy'],
            'train_and': train_params['and_sell'],
            'train_or': train_params['or_sell'],
            'train_return': best_params['train_return'],
            'train_trades': int(best_params['train_trades']),
            'test_buy': test_params['buy'],
            'test_and': test_params['and_sell'],
            'test_or': test_params['or_sell'],
            'test_return': test_return,
            'test_trades': len(test_trades),
            'position_carry': end_position > 0,
            'trades_detail': test_trades
        })

    return all_results


def main():
    print("=" * 80)
    print("Walk-Forward 详细分析 - 美股七姐妹")
    print("=" * 80)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.dirname(__file__)
    output_dir = os.path.join(base_dir, f'analysis_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n输出目录: {output_dir}")

    all_results = []

    for symbol in SYMBOLS:
        results = analyze_symbol(symbol, output_dir)
        if results:
            all_results.extend(results)

    # 保存汇总数据
    if all_results:
        # 移除 trades_detail 列（太长）
        summary_data = [{k: v for k, v in r.items() if k != 'trades_detail'} for r in all_results]
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(output_dir, 'analysis_summary.csv'), index=False)

        # 保存详细交易记录
        trades_records = []
        for r in all_results:
            for t in r['trades_detail']:
                trades_records.append({
                    'symbol': r['symbol'],
                    'window': r['window'],
                    'type': t['type'],
                    'date': t['date'].strftime('%Y-%m-%d'),
                    'price': t['price'],
                    'sentiment': t['sentiment'],
                    'reason': t['reason'],
                    'profit_pct': t.get('profit_pct', None)
                })

        if trades_records:
            trades_df = pd.DataFrame(trades_records)
            trades_df.to_csv(os.path.join(output_dir, 'trades_detail.csv'), index=False)

        # 生成汇总报告
        with open(os.path.join(output_dir, 'analysis_report.txt'), 'w') as f:
            f.write("Walk-Forward 详细分析报告 - 美股七姐妹\n")
            f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")

            for symbol in SYMBOLS:
                symbol_results = [r for r in all_results if r['symbol'] == symbol]
                if not symbol_results:
                    continue

                f.write(f"\n{'='*70}\n")
                f.write(f"{symbol}\n")
                f.write(f"{'='*70}\n\n")

                cumret = 1
                for r in symbol_results:
                    cumret *= (1 + r['test_return'] / 100)

                    f.write(f"{r['window']} ({r['test_year']}):\n")
                    f.write(f"  训练期参数: Buy<{r['train_buy']}, AND>{r['train_and']}, OR>{r['train_or']}\n")
                    f.write(f"  训练期收益: {r['train_return']:+.1f}% ({r['train_trades']}次)\n")
                    f.write(f"  测试期参数: Buy<{r['test_buy']:.0f}, AND>{r['test_and']}, OR>{r['test_or']}\n")
                    f.write(f"  测试期收益: {r['test_return']:+.1f}% ({r['test_trades']}次)\n")
                    f.write(f"  持仓延续: {'是' if r['position_carry'] else '否'}\n")

                    if r['trades_detail']:
                        f.write(f"  交易记录:\n")
                        for t in r['trades_detail']:
                            date_str = t['date'].strftime('%Y-%m-%d')
                            if t['type'] == 'BUY':
                                f.write(f"    {date_str} BUY  @ ${t['price']:.2f} | Sentiment: {t['sentiment']:.1f}\n")
                            else:
                                f.write(f"    {date_str} SELL @ ${t['price']:.2f} | Sentiment: {t['sentiment']:.1f} | {t.get('profit_pct', 0):+.1f}%\n")
                    f.write("\n")

                f.write(f"累计测试期收益: {(cumret-1)*100:+.1f}%\n")

        print(f"\n{'='*80}")
        print("完成!")
        print(f"{'='*80}")
        print(f"\n输出文件:")
        print(f"  {output_dir}/")
        print(f"    ├── analysis_summary.csv      # 汇总数据")
        print(f"    ├── trades_detail.csv         # 详细交易记录")
        print(f"    ├── analysis_report.txt       # 文本报告")
        print(f"    └── {{SYMBOL}}_{{Window}}_analysis.png  # 可视化图表")

    return all_results


if __name__ == "__main__":
    main()
