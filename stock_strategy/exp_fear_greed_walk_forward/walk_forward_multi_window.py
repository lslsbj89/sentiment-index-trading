"""
Multi-Window Walk-Forward Validation
多窗口滚动验证，评估参数在不同市场环境下的稳健性

设计：
- 6个滚动窗口 (2016-2025)
- 每个窗口: 训练4年 + 测试1年
- 对比: 夏普比率排序 vs 综合评分排序
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

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

from data_loader import DataLoader
from backtest_engine import EnhancedBacktester

# ============================================================
# 配置
# ============================================================
SYMBOL = "TSLA"  # 目标股票

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
BACKTEST_PARAMS = {
    "initial_capital": 100000,
    "commission_rate": 0.001,
    "slippage_rate": 0.001,
    "take_profit_pct": 999.0,
    "stop_loss_pct": 999.0,
    "max_holding_days": 999,
    "position_pct": 0.8
}

# Walk-Forward 窗口定义 (训练4年 + 测试1年)
WINDOWS = [
    {"name": "W1", "train": ("2016-01-01", "2019-12-31"), "test": ("2020-01-01", "2020-12-31")},
    {"name": "W2", "train": ("2017-01-01", "2020-12-31"), "test": ("2021-01-01", "2021-12-31")},
    {"name": "W3", "train": ("2018-01-01", "2021-12-31"), "test": ("2022-01-01", "2022-12-31")},
    {"name": "W4", "train": ("2019-01-01", "2022-12-31"), "test": ("2023-01-01", "2023-12-31")},
    {"name": "W5", "train": ("2020-01-01", "2023-12-31"), "test": ("2024-01-01", "2024-12-31")},
    {"name": "W6", "train": ("2021-01-01", "2024-12-31"), "test": ("2025-01-01", "2025-12-31")},
]


def load_fear_greed_index(symbol):
    """加载情绪指数"""
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
    df['smoothed_index'] = df['smoothed_index'].astype(float)
    return df


def run_backtest(price_data, sentiment_data, buy_threshold, and_sell_threshold, or_threshold):
    """运行单次回测"""
    signals = pd.DataFrame(index=price_data.index)
    signals['smoothed_index'] = sentiment_data['smoothed_index'].reindex(price_data.index)
    signals['Close'] = price_data['Close']
    signals['MA50'] = price_data['Close'].rolling(window=50).mean()

    signals['buy_signal'] = (signals['smoothed_index'] < buy_threshold).astype(int)

    and_condition = (signals['smoothed_index'] > and_sell_threshold) & (signals['Close'] < signals['MA50'])
    or_condition = signals['smoothed_index'] > or_threshold
    signals['sell_signal'] = (and_condition | or_condition).astype(int)

    signals['prob_profit'] = 0.5
    signals['position_size'] = 0

    backtester = EnhancedBacktester(**BACKTEST_PARAMS, use_dynamic_position=True)
    portfolio, metrics, trades = backtester.run_backtest_with_sell_signal(signals, price_data)

    return portfolio, metrics, trades, signals


def grid_search(price_data, sentiment_data, verbose=False):
    """网格搜索"""
    results = []
    total = len(BUY_THRESHOLDS) * len(AND_SELL_THRESHOLDS) * len(OR_THRESHOLDS)

    for buy_t, and_t, or_t in product(BUY_THRESHOLDS, AND_SELL_THRESHOLDS, OR_THRESHOLDS):
        try:
            portfolio, metrics, trades, _ = run_backtest(
                price_data, sentiment_data, buy_t, and_t, or_t
            )

            results.append({
                'buy_threshold': buy_t,
                'and_sell_threshold': and_t,
                'or_threshold': or_t,
                'total_return': metrics.get('total_return', 0) * 100,
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0) * 100,
                'num_trades': len(trades),
                'win_rate': metrics.get('trade_win_rate', 0) * 100
            })
        except:
            pass

    return pd.DataFrame(results)


def compute_composite_score(row):
    """综合评分"""
    sharpe = row['sharpe_ratio']
    max_dd = abs(row['max_drawdown'])
    win_rate = row['win_rate'] / 100
    num_trades = row['num_trades']

    if num_trades < 2:
        trade_score = 0.3
    elif num_trades > 30:
        trade_score = 0.5
    else:
        trade_score = 1.0

    score = (
        0.4 * sharpe +
        0.3 * (1 - max_dd / 100) +
        0.2 * win_rate +
        0.1 * trade_score
    )

    return score


def run_single_window(window, price_data, sentiment_data):
    """运行单个窗口的walk-forward验证"""
    train_start = pd.Timestamp(window['train'][0], tz='UTC')
    train_end = pd.Timestamp(window['train'][1], tz='UTC')
    test_start = pd.Timestamp(window['test'][0], tz='UTC')
    test_end = pd.Timestamp(window['test'][1], tz='UTC')

    # 分割数据
    train_price = price_data[(price_data.index >= train_start) & (price_data.index <= train_end)]
    test_price = price_data[(price_data.index >= test_start) & (price_data.index <= test_end)]
    train_sentiment = sentiment_data[(sentiment_data.index >= train_start) & (sentiment_data.index <= train_end)]
    test_sentiment = sentiment_data[(sentiment_data.index >= test_start) & (sentiment_data.index <= test_end)]

    if len(train_price) < 100 or len(test_price) < 50:
        return None

    # 网格搜索
    train_results = grid_search(train_price, train_sentiment)
    if len(train_results) == 0:
        return None

    train_results['composite_score'] = train_results.apply(compute_composite_score, axis=1)

    # 方法1: 夏普比率排序
    by_sharpe = train_results.sort_values('sharpe_ratio', ascending=False).iloc[0]
    params_sharpe = {
        'buy': int(by_sharpe['buy_threshold']),
        'and_sell': int(by_sharpe['and_sell_threshold']),
        'or': int(by_sharpe['or_threshold'])
    }

    # 方法2: 综合评分排序
    by_composite = train_results.sort_values('composite_score', ascending=False).iloc[0]
    params_composite = {
        'buy': int(by_composite['buy_threshold']),
        'and_sell': int(by_composite['and_sell_threshold']),
        'or': int(by_composite['or_threshold'])
    }

    # 测试期验证
    _, test_metrics_sharpe, test_trades_sharpe, _ = run_backtest(
        test_price, test_sentiment,
        params_sharpe['buy'], params_sharpe['and_sell'], params_sharpe['or']
    )

    _, test_metrics_composite, test_trades_composite, _ = run_backtest(
        test_price, test_sentiment,
        params_composite['buy'], params_composite['and_sell'], params_composite['or']
    )

    return {
        'window': window['name'],
        'train_period': f"{window['train'][0][:4]}-{window['train'][1][:4]}",
        'test_year': window['test'][0][:4],
        # 夏普方法
        'sharpe_params': params_sharpe,
        'sharpe_train_return': by_sharpe['total_return'],
        'sharpe_train_sharpe': by_sharpe['sharpe_ratio'],
        'sharpe_train_winrate': by_sharpe['win_rate'],
        'sharpe_test_return': test_metrics_sharpe['total_return'] * 100,
        'sharpe_test_sharpe': test_metrics_sharpe['sharpe_ratio'],
        'sharpe_test_trades': len(test_trades_sharpe),
        'sharpe_test_winrate': test_metrics_sharpe.get('trade_win_rate', 0) * 100,
        # 综合评分方法
        'composite_params': params_composite,
        'composite_train_return': by_composite['total_return'],
        'composite_train_sharpe': by_composite['sharpe_ratio'],
        'composite_train_winrate': by_composite['win_rate'],
        'composite_test_return': test_metrics_composite['total_return'] * 100,
        'composite_test_sharpe': test_metrics_composite['sharpe_ratio'],
        'composite_test_trades': len(test_trades_composite),
        'composite_test_winrate': test_metrics_composite.get('trade_win_rate', 0) * 100,
    }


def visualize_results(results_df, symbol):
    """可视化多窗口结果"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    windows = results_df['test_year'].tolist()
    x = np.arange(len(windows))
    width = 0.35

    # 1. 测试期收益对比
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x - width/2, results_df['sharpe_test_return'], width, label='Sharpe-Only', color='steelblue')
    bars2 = ax1.bar(x + width/2, results_df['composite_test_return'], width, label='Composite', color='darkorange')
    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax1.set_ylabel('Test Return (%)')
    ax1.set_title('Test Period Return by Window')
    ax1.set_xticks(x)
    ax1.set_xticklabels(windows)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 标注数值
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    # 2. 测试期夏普对比
    ax2 = axes[0, 1]
    ax2.bar(x - width/2, results_df['sharpe_test_sharpe'], width, label='Sharpe-Only', color='steelblue')
    ax2.bar(x + width/2, results_df['composite_test_sharpe'], width, label='Composite', color='darkorange')
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax2.axhline(y=1, color='green', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_ylabel('Test Sharpe Ratio')
    ax2.set_title('Test Period Sharpe Ratio by Window')
    ax2.set_xticks(x)
    ax2.set_xticklabels(windows)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 累计收益对比
    ax3 = axes[1, 0]
    sharpe_cumret = (1 + results_df['sharpe_test_return']/100).cumprod() * 100 - 100
    composite_cumret = (1 + results_df['composite_test_return']/100).cumprod() * 100 - 100
    ax3.plot(windows, sharpe_cumret, 'o-', label='Sharpe-Only', color='steelblue', linewidth=2, markersize=8)
    ax3.plot(windows, composite_cumret, 's-', label='Composite', color='darkorange', linewidth=2, markersize=8)
    ax3.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax3.set_ylabel('Cumulative Return (%)')
    ax3.set_title('Cumulative Test Return (2020-2025)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 标注最终值
    ax3.annotate(f'{sharpe_cumret.iloc[-1]:.1f}%', xy=(windows[-1], sharpe_cumret.iloc[-1]),
                xytext=(5, 0), textcoords="offset points", fontsize=10, fontweight='bold', color='steelblue')
    ax3.annotate(f'{composite_cumret.iloc[-1]:.1f}%', xy=(windows[-1], composite_cumret.iloc[-1]),
                xytext=(5, -15), textcoords="offset points", fontsize=10, fontweight='bold', color='darkorange')

    # 4. 汇总统计
    ax4 = axes[1, 1]
    ax4.axis('off')

    # 计算统计
    sharpe_wins = sum(results_df['sharpe_test_return'] > results_df['composite_test_return'])
    composite_wins = sum(results_df['composite_test_return'] > results_df['sharpe_test_return'])
    ties = sum(results_df['sharpe_test_return'] == results_df['composite_test_return'])

    sharpe_avg_ret = results_df['sharpe_test_return'].mean()
    composite_avg_ret = results_df['composite_test_return'].mean()
    sharpe_avg_sharpe = results_df['sharpe_test_sharpe'].mean()
    composite_avg_sharpe = results_df['composite_test_sharpe'].mean()

    sharpe_positive = sum(results_df['sharpe_test_return'] > 0)
    composite_positive = sum(results_df['composite_test_return'] > 0)

    summary_text = f"""
    +------------------------------------------------------------------+
    |  {symbol} Multi-Window Walk-Forward Summary (2020-2025)          |
    +------------------------------------------------------------------+
    |                           Sharpe-Only      Composite             |
    |  ----------------------------------------------------------------|
    |  Windows Won:              {sharpe_wins:>6}          {composite_wins:>6}                |
    |  Ties:                                  {ties}                       |
    |  ----------------------------------------------------------------|
    |  Avg Test Return:          {sharpe_avg_ret:>+6.2f}%         {composite_avg_ret:>+6.2f}%              |
    |  Avg Test Sharpe:          {sharpe_avg_sharpe:>6.2f}          {composite_avg_sharpe:>6.2f}               |
    |  Profitable Windows:       {sharpe_positive:>6}/6          {composite_positive:>6}/6               |
    |  ----------------------------------------------------------------|
    |  Cumulative Return:        {sharpe_cumret.iloc[-1]:>+6.1f}%         {composite_cumret.iloc[-1]:>+6.1f}%              |
    +------------------------------------------------------------------+
    """

    ax4.text(0.5, 0.5, summary_text, transform=ax4.transAxes,
             fontsize=11, fontfamily='monospace',
             verticalalignment='center', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    fig.suptitle(f'{symbol} Walk-Forward Validation: Sharpe-Only vs Composite Scoring',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    output_file = os.path.join(os.path.dirname(__file__), f'walk_forward_multi_{symbol}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_file


def main():
    print("=" * 80)
    print(f"Multi-Window Walk-Forward Validation: {SYMBOL}")
    print("=" * 80)
    print(f"\n共 {len(WINDOWS)} 个窗口:")
    for w in WINDOWS:
        print(f"  {w['name']}: Train {w['train'][0][:4]}-{w['train'][1][:4]} → Test {w['test'][0][:4]}")

    # 加载数据
    print("\n加载数据...")
    loader = DataLoader(db_config)
    price_data = loader.load_ohlcv(SYMBOL, start_date="2015-01-01")
    loader.close()

    sentiment_data = load_fear_greed_index(SYMBOL)

    print(f"  价格数据: {len(price_data)} 行 ({price_data.index.min().strftime('%Y-%m-%d')} ~ {price_data.index.max().strftime('%Y-%m-%d')})")
    print(f"  情绪数据: {len(sentiment_data)} 行")

    # 运行所有窗口
    results = []
    for window in WINDOWS:
        print(f"\n{'='*60}")
        print(f"{window['name']}: Train {window['train'][0][:4]}-{window['train'][1][:4]} → Test {window['test'][0][:4]}")
        print("=" * 60)

        result = run_single_window(window, price_data, sentiment_data)

        if result:
            print(f"\n  Sharpe-Only:  params=({result['sharpe_params']['buy']}, {result['sharpe_params']['and_sell']}, {result['sharpe_params']['or']})")
            print(f"                train: {result['sharpe_train_return']:.1f}% | test: {result['sharpe_test_return']:+.1f}%")

            print(f"\n  Composite:    params=({result['composite_params']['buy']}, {result['composite_params']['and_sell']}, {result['composite_params']['or']})")
            print(f"                train: {result['composite_train_return']:.1f}% | test: {result['composite_test_return']:+.1f}%")

            # 判断赢家
            if result['sharpe_test_return'] > result['composite_test_return']:
                print(f"\n  Winner: Sharpe-Only (+{result['sharpe_test_return']-result['composite_test_return']:.1f}%)")
            elif result['composite_test_return'] > result['sharpe_test_return']:
                print(f"\n  Winner: Composite (+{result['composite_test_return']-result['sharpe_test_return']:.1f}%)")
            else:
                print(f"\n  Tie!")

            results.append(result)
        else:
            print("  ⚠️ 数据不足，跳过")

    # 汇总结果
    if results:
        results_df = pd.DataFrame(results)

        print("\n" + "=" * 80)
        print("汇总结果")
        print("=" * 80)

        print(f"\n{'Window':<8} {'Test':<6} {'Sharpe-Only':>15} {'Composite':>15} {'Winner':>12}")
        print("-" * 60)
        for _, row in results_df.iterrows():
            s_ret = row['sharpe_test_return']
            c_ret = row['composite_test_return']
            if s_ret > c_ret:
                winner = "Sharpe"
            elif c_ret > s_ret:
                winner = "Composite"
            else:
                winner = "Tie"
            print(f"{row['window']:<8} {row['test_year']:<6} {s_ret:>+14.1f}% {c_ret:>+14.1f}% {winner:>12}")

        # 统计
        sharpe_wins = sum(results_df['sharpe_test_return'] > results_df['composite_test_return'])
        composite_wins = sum(results_df['composite_test_return'] > results_df['sharpe_test_return'])

        print("-" * 60)
        print(f"{'Avg':>14} {results_df['sharpe_test_return'].mean():>+14.1f}% {results_df['composite_test_return'].mean():>+14.1f}%")

        # 累计收益
        sharpe_cumret = (1 + results_df['sharpe_test_return']/100).prod() - 1
        composite_cumret = (1 + results_df['composite_test_return']/100).prod() - 1
        print(f"{'Cumulative':>14} {sharpe_cumret*100:>+14.1f}% {composite_cumret*100:>+14.1f}%")

        print(f"\n胜负统计: Sharpe {sharpe_wins} : {composite_wins} Composite")

        # 生成可视化
        viz_file = visualize_results(results_df, SYMBOL)

        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_dir = os.path.dirname(__file__)

        results_df.to_csv(os.path.join(base_dir, f'walk_forward_multi_{SYMBOL}_{timestamp}.csv'), index=False)

        # 保存最优参数
        winner_method = "Composite" if composite_cumret > sharpe_cumret else "Sharpe-Only"
        with open(os.path.join(base_dir, f'best_params_{SYMBOL}_multi.txt'), 'w') as f:
            f.write(f"实验: {SYMBOL} Multi-Window Walk-Forward\n")
            f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"窗口数: {len(results)}\n\n")

            f.write("=" * 60 + "\n")
            f.write("各窗口结果\n")
            f.write("=" * 60 + "\n\n")

            for _, row in results_df.iterrows():
                f.write(f"{row['window']} ({row['test_year']}):\n")
                f.write(f"  Sharpe:    ({row['sharpe_params']['buy']}, {row['sharpe_params']['and_sell']}, {row['sharpe_params']['or']}) → {row['sharpe_test_return']:+.1f}%\n")
                f.write(f"  Composite: ({row['composite_params']['buy']}, {row['composite_params']['and_sell']}, {row['composite_params']['or']}) → {row['composite_test_return']:+.1f}%\n\n")

            f.write("=" * 60 + "\n")
            f.write("汇总\n")
            f.write("=" * 60 + "\n")
            f.write(f"Sharpe-Only 累计收益: {sharpe_cumret*100:+.1f}%\n")
            f.write(f"Composite 累计收益:   {composite_cumret*100:+.1f}%\n")
            f.write(f"胜负: Sharpe {sharpe_wins} : {composite_wins} Composite\n")
            f.write(f"\n推荐方法: {winner_method}\n")

        print(f"\n" + "=" * 80)
        print("✅ Multi-Window Walk-Forward 验证完成!")
        print("=" * 80)
        print(f"\n已保存文件:")
        print(f"  - walk_forward_multi_{SYMBOL}_{timestamp}.csv")
        print(f"  - best_params_{SYMBOL}_multi.txt")
        print(f"  - {os.path.basename(viz_file)}")

        return results_df


if __name__ == "__main__":
    main()
