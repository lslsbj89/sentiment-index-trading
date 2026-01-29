"""
Walk-Forward 网格搜索验证
避免过拟合，真实评估参数泛化能力

设计：
- 训练期：4年 (2021-2024)
- 测试期：1年 (2025)
- 用训练期最优参数在测试期验证
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

# Walk-Forward 时间窗口
TRAIN_START = "2021-01-01"
TRAIN_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2025-12-31"


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
    # 构建信号
    signals = pd.DataFrame(index=price_data.index)
    signals['smoothed_index'] = sentiment_data['smoothed_index'].reindex(price_data.index)
    signals['Close'] = price_data['Close']
    signals['MA50'] = price_data['Close'].rolling(window=50).mean()

    # 买入信号
    signals['buy_signal'] = (signals['smoothed_index'] < buy_threshold).astype(int)

    # 卖出信号
    and_condition = (signals['smoothed_index'] > and_sell_threshold) & (signals['Close'] < signals['MA50'])
    or_condition = signals['smoothed_index'] > or_threshold
    signals['sell_signal'] = (and_condition | or_condition).astype(int)

    signals['prob_profit'] = 0.5
    signals['position_size'] = 0

    # 回测
    backtester = EnhancedBacktester(**BACKTEST_PARAMS, use_dynamic_position=True)
    portfolio, metrics, trades = backtester.run_backtest_with_sell_signal(signals, price_data)

    return portfolio, metrics, trades, signals


def grid_search(price_data, sentiment_data, verbose=True):
    """网格搜索找最优参数"""
    results = []
    total = len(BUY_THRESHOLDS) * len(AND_SELL_THRESHOLDS) * len(OR_THRESHOLDS)
    count = 0

    for buy_t, and_t, or_t in product(BUY_THRESHOLDS, AND_SELL_THRESHOLDS, OR_THRESHOLDS):
        count += 1

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

            if verbose and count % 20 == 0:
                print(f"  进度: {count}/{total}", end='\r')

        except Exception as e:
            pass

    if verbose:
        print(f"  搜索完成: {len(results)} 个有效结果")

    return pd.DataFrame(results)


def compute_composite_score(row):
    """
    综合评分函数

    权重分配:
    - 夏普比率: 40% (风险调整后收益)
    - 回撤控制: 30% (1 - |max_dd|/100)
    - 胜率: 20%
    - 交易频率: 10% (避免过度或过少交易)
    """
    sharpe = row['sharpe_ratio']
    max_dd = abs(row['max_drawdown'])
    win_rate = row['win_rate'] / 100  # 转为0-1
    num_trades = row['num_trades']

    # 交易频率评分 (理想: 5-20次/4年)
    if num_trades < 2:
        trade_score = 0.3  # 交易太少
    elif num_trades > 30:
        trade_score = 0.5  # 交易太多
    else:
        trade_score = 1.0  # 合理范围

    # 综合评分
    score = (
        0.4 * sharpe +
        0.3 * (1 - max_dd / 100) +
        0.2 * win_rate +
        0.1 * trade_score
    )

    return score


def visualize_comparison(train_result, test_result, symbol, best_params, method_name=""):
    """可视化训练期 vs 测试期对比"""
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 2, height_ratios=[3, 2, 2, 1.5], hspace=0.3, wspace=0.25,
                          left=0.06, right=0.94, top=0.93, bottom=0.05)

    # ========================================
    # 子图1: 训练期价格 + 交易
    # ========================================
    ax1 = fig.add_subplot(gs[0, 0])

    train_price = train_result['price_data']
    train_trades = train_result['trades']

    ax1.plot(train_price.index, train_price['Close'], 'b-', linewidth=1.5, alpha=0.8)
    ax1.plot(train_price.index, train_price['Close'].rolling(50).mean(), 'orange', linewidth=1, alpha=0.7)

    for trade in train_trades:
        color = 'lightgreen' if trade['profit'] > 0 else 'lightcoral'
        ax1.axvspan(trade['entry_date'], trade['exit_date'], alpha=0.2, color=color)
        ax1.scatter(trade['entry_date'], trade['entry_price'], marker='^', s=150, c='green', zorder=5)
        ax1.scatter(trade['exit_date'], trade['exit_price'], marker='v', s=150, c='red', zorder=5)

    ax1.set_title(f'TRAIN: {TRAIN_START[:4]}-{TRAIN_END[:4]} | Return: {train_result["metrics"]["total_return"]*100:.1f}% | Sharpe: {train_result["metrics"]["sharpe_ratio"]:.2f}',
                  fontsize=12, fontweight='bold')
    ax1.set_ylabel('Price ($)')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # ========================================
    # 子图2: 测试期价格 + 交易
    # ========================================
    ax2 = fig.add_subplot(gs[0, 1])

    test_price = test_result['price_data']
    test_trades = test_result['trades']

    ax2.plot(test_price.index, test_price['Close'], 'b-', linewidth=1.5, alpha=0.8)
    ax2.plot(test_price.index, test_price['Close'].rolling(50).mean(), 'orange', linewidth=1, alpha=0.7)

    for trade in test_trades:
        color = 'lightgreen' if trade['profit'] > 0 else 'lightcoral'
        ax2.axvspan(trade['entry_date'], trade['exit_date'], alpha=0.2, color=color)
        ax2.scatter(trade['entry_date'], trade['entry_price'], marker='^', s=150, c='green', zorder=5)
        ax2.scatter(trade['exit_date'], trade['exit_price'], marker='v', s=150, c='red', zorder=5)

    ax2.set_title(f'TEST: {TEST_START[:4]} | Return: {test_result["metrics"]["total_return"]*100:.1f}% | Sharpe: {test_result["metrics"]["sharpe_ratio"]:.2f}',
                  fontsize=12, fontweight='bold')
    ax2.set_ylabel('Price ($)')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # ========================================
    # 子图3: 训练期情绪指数
    # ========================================
    ax3 = fig.add_subplot(gs[1, 0])

    train_sentiment = train_result['signals']['smoothed_index']
    ax3.fill_between(train_sentiment.index, 0, train_sentiment,
                     where=train_sentiment > 0, color='lightcoral', alpha=0.5)
    ax3.fill_between(train_sentiment.index, 0, train_sentiment,
                     where=train_sentiment <= 0, color='lightgreen', alpha=0.5)
    ax3.plot(train_sentiment.index, train_sentiment, 'k-', linewidth=1)

    ax3.axhline(y=best_params['buy'], color='green', linestyle='--', linewidth=2,
                label=f'Buy < {best_params["buy"]}')
    ax3.axhline(y=best_params['and_sell'], color='orange', linestyle='--', linewidth=2,
                label=f'AND > {best_params["and_sell"]}')
    ax3.axhline(y=best_params['or'], color='red', linestyle='--', linewidth=2,
                label=f'OR > {best_params["or"]}')

    ax3.set_ylabel('Sentiment Index')
    ax3.set_ylim(-60, 80)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # ========================================
    # 子图4: 测试期情绪指数
    # ========================================
    ax4 = fig.add_subplot(gs[1, 1])

    test_sentiment = test_result['signals']['smoothed_index']
    ax4.fill_between(test_sentiment.index, 0, test_sentiment,
                     where=test_sentiment > 0, color='lightcoral', alpha=0.5)
    ax4.fill_between(test_sentiment.index, 0, test_sentiment,
                     where=test_sentiment <= 0, color='lightgreen', alpha=0.5)
    ax4.plot(test_sentiment.index, test_sentiment, 'k-', linewidth=1)

    ax4.axhline(y=best_params['buy'], color='green', linestyle='--', linewidth=2)
    ax4.axhline(y=best_params['and_sell'], color='orange', linestyle='--', linewidth=2)
    ax4.axhline(y=best_params['or'], color='red', linestyle='--', linewidth=2)

    ax4.set_ylabel('Sentiment Index')
    ax4.set_ylim(-60, 80)
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # ========================================
    # 子图5: 训练期组合价值
    # ========================================
    ax5 = fig.add_subplot(gs[2, 0])

    train_portfolio = train_result['portfolio']
    ax5.fill_between(train_portfolio.index, 100000, train_portfolio['total_value'],
                     where=train_portfolio['total_value'] >= 100000, color='lightgreen', alpha=0.5)
    ax5.fill_between(train_portfolio.index, 100000, train_portfolio['total_value'],
                     where=train_portfolio['total_value'] < 100000, color='lightcoral', alpha=0.5)
    ax5.plot(train_portfolio.index, train_portfolio['total_value'], 'b-', linewidth=1.5)
    ax5.axhline(y=100000, color='gray', linestyle='--', linewidth=1)

    final_train = train_portfolio['total_value'].iloc[-1]
    ax5.annotate(f'${final_train:,.0f}', xy=(train_portfolio.index[-1], final_train),
                xytext=(-60, 10), textcoords='offset points', fontsize=10, fontweight='bold')

    ax5.set_ylabel('Portfolio Value ($)')
    ax5.grid(True, alpha=0.3)
    ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # ========================================
    # 子图6: 测试期组合价值
    # ========================================
    ax6 = fig.add_subplot(gs[2, 1])

    test_portfolio = test_result['portfolio']
    ax6.fill_between(test_portfolio.index, 100000, test_portfolio['total_value'],
                     where=test_portfolio['total_value'] >= 100000, color='lightgreen', alpha=0.5)
    ax6.fill_between(test_portfolio.index, 100000, test_portfolio['total_value'],
                     where=test_portfolio['total_value'] < 100000, color='lightcoral', alpha=0.5)
    ax6.plot(test_portfolio.index, test_portfolio['total_value'], 'b-', linewidth=1.5)
    ax6.axhline(y=100000, color='gray', linestyle='--', linewidth=1)

    final_test = test_portfolio['total_value'].iloc[-1]
    ax6.annotate(f'${final_test:,.0f}', xy=(test_portfolio.index[-1], final_test),
                xytext=(-60, 10), textcoords='offset points', fontsize=10, fontweight='bold')

    ax6.set_ylabel('Portfolio Value ($)')
    ax6.grid(True, alpha=0.3)
    ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    ax6.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # ========================================
    # 子图7: 对比摘要
    # ========================================
    ax7 = fig.add_subplot(gs[3, :])
    ax7.axis('off')

    # 计算变化
    return_change = test_result['metrics']['total_return']*100 - train_result['metrics']['total_return']*100
    sharpe_change = test_result['metrics']['sharpe_ratio'] - train_result['metrics']['sharpe_ratio']

    # 判断是否过拟合
    is_overfit = test_result['metrics']['total_return'] < train_result['metrics']['total_return'] * 0.3
    overfit_status = "WARNING: Possible Overfitting" if is_overfit else "GOOD: Generalization OK"

    summary_text = f"""
    +--------------------------------------------------------------------------------------------------------------+
    |  {symbol} Walk-Forward Validation Results                                                                    |
    |  ============================================================================================================|
    |  Best Params: buy < {best_params['buy']}, AND > {best_params['and_sell']} & < MA50, OR > {best_params['or']}                                                       |
    |  ------------------------------------------------------------------------------------------------------------|
    |                          TRAIN ({TRAIN_START[:4]}-{TRAIN_END[:4]})              TEST ({TEST_START[:4]})                     Change            |
    |  ------------------------------------------------------------------------------------------------------------|
    |  Total Return:           {train_result['metrics']['total_return']*100:>+8.2f}%                      {test_result['metrics']['total_return']*100:>+8.2f}%                  {return_change:>+6.2f}%           |
    |  Sharpe Ratio:           {train_result['metrics']['sharpe_ratio']:>8.2f}                       {test_result['metrics']['sharpe_ratio']:>8.2f}                   {sharpe_change:>+6.2f}            |
    |  Max Drawdown:           {train_result['metrics']['max_drawdown']*100:>8.2f}%                      {test_result['metrics']['max_drawdown']*100:>8.2f}%                                    |
    |  Num Trades:             {len(train_trades):>8}                       {len(test_trades):>8}                                       |
    |  Win Rate:               {train_result['metrics'].get('trade_win_rate', 0)*100:>7.1f}%                      {test_result['metrics'].get('trade_win_rate', 0)*100:>7.1f}%                                    |
    |  ------------------------------------------------------------------------------------------------------------|
    |  Status: {overfit_status}                                                                                    |
    +--------------------------------------------------------------------------------------------------------------+
    """

    ax7.text(0.5, 0.5, summary_text, transform=ax7.transAxes,
             fontsize=10, fontfamily='monospace',
             verticalalignment='center', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    title_suffix = f" ({method_name})" if method_name else ""
    fig.suptitle(f'{symbol} Walk-Forward Validation{title_suffix}: Train {TRAIN_START[:4]}-{TRAIN_END[:4]} → Test {TEST_START[:4]}',
                 fontsize=14, fontweight='bold', y=0.98)

    # 保存
    filename_suffix = f"_{method_name}" if method_name else ""
    output_file = os.path.join(os.path.dirname(__file__), f'walk_forward_{symbol}{filename_suffix}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_file


def test_with_params(train_price, train_sentiment, test_price, test_sentiment, params, method_name):
    """用指定参数进行训练和测试"""
    # 训练期回测
    train_portfolio, train_metrics, train_trades, train_signals = run_backtest(
        train_price, train_sentiment,
        params['buy'], params['and_sell'], params['or']
    )

    # 测试期回测
    test_portfolio, test_metrics, test_trades, test_signals = run_backtest(
        test_price, test_sentiment,
        params['buy'], params['and_sell'], params['or']
    )

    return {
        'method': method_name,
        'params': params,
        'train': {
            'portfolio': train_portfolio,
            'metrics': train_metrics,
            'trades': train_trades,
            'signals': train_signals,
            'price_data': train_price
        },
        'test': {
            'portfolio': test_portfolio,
            'metrics': test_metrics,
            'trades': test_trades,
            'signals': test_signals,
            'price_data': test_price
        }
    }


def main():
    print("=" * 80)
    print(f"Walk-Forward 网格搜索验证: {SYMBOL}")
    print("对比: 夏普比率排序 vs 综合评分排序")
    print("=" * 80)
    print(f"\n训练期: {TRAIN_START} ~ {TRAIN_END} (4年)")
    print(f"测试期: {TEST_START} ~ {TEST_END} (1年)")
    print(f"\n参数搜索空间:")
    print(f"  买入阈值: {BUY_THRESHOLDS}")
    print(f"  AND卖出阈值: {AND_SELL_THRESHOLDS}")
    print(f"  OR兜底阈值: {OR_THRESHOLDS}")
    print(f"  总组合数: {len(BUY_THRESHOLDS) * len(AND_SELL_THRESHOLDS) * len(OR_THRESHOLDS)}")

    # 加载数据
    print("\n加载数据...")
    loader = DataLoader(db_config)
    price_data = loader.load_ohlcv(SYMBOL, start_date="2020-01-01")
    loader.close()

    sentiment_data = load_fear_greed_index(SYMBOL)

    print(f"  价格数据: {len(price_data)} 行")
    print(f"  情绪数据: {len(sentiment_data)} 行")

    # 分割训练/测试数据
    train_start_ts = pd.Timestamp(TRAIN_START, tz='UTC')
    train_end_ts = pd.Timestamp(TRAIN_END, tz='UTC')
    test_start_ts = pd.Timestamp(TEST_START, tz='UTC')
    test_end_ts = pd.Timestamp(TEST_END, tz='UTC')

    train_price = price_data[(price_data.index >= train_start_ts) & (price_data.index <= train_end_ts)]
    test_price = price_data[(price_data.index >= test_start_ts) & (price_data.index <= test_end_ts)]

    train_sentiment = sentiment_data[(sentiment_data.index >= train_start_ts) & (sentiment_data.index <= train_end_ts)]
    test_sentiment = sentiment_data[(sentiment_data.index >= test_start_ts) & (sentiment_data.index <= test_end_ts)]

    print(f"\n训练期数据: {len(train_price)} 天")
    print(f"测试期数据: {len(test_price)} 天")

    # ========================================
    # 阶段1: 训练期网格搜索
    # ========================================
    print("\n" + "=" * 70)
    print("阶段1: 训练期网格搜索 (2021-2024)")
    print("=" * 70)

    train_results = grid_search(train_price, train_sentiment)

    # 计算综合评分
    train_results['composite_score'] = train_results.apply(compute_composite_score, axis=1)

    # ========================================
    # 方法1: 按夏普比率排序
    # ========================================
    print("\n" + "-" * 70)
    print("方法1: 按夏普比率排序")
    print("-" * 70)

    by_sharpe = train_results.sort_values('sharpe_ratio', ascending=False).iloc[0]
    params_sharpe = {
        'buy': int(by_sharpe['buy_threshold']),
        'and_sell': int(by_sharpe['and_sell_threshold']),
        'or': int(by_sharpe['or_threshold'])
    }

    print(f"  最优参数: buy < {params_sharpe['buy']}, AND > {params_sharpe['and_sell']}, OR > {params_sharpe['or']}")
    print(f"  训练期: 收益 {by_sharpe['total_return']:.1f}% | 夏普 {by_sharpe['sharpe_ratio']:.2f} | 回撤 {by_sharpe['max_drawdown']:.1f}% | 交易 {int(by_sharpe['num_trades'])} | 胜率 {by_sharpe['win_rate']:.0f}%")
    print(f"  综合评分: {by_sharpe['composite_score']:.4f}")

    # ========================================
    # 方法2: 按综合评分排序
    # ========================================
    print("\n" + "-" * 70)
    print("方法2: 按综合评分排序 (夏普40% + 回撤30% + 胜率20% + 频率10%)")
    print("-" * 70)

    by_composite = train_results.sort_values('composite_score', ascending=False).iloc[0]
    params_composite = {
        'buy': int(by_composite['buy_threshold']),
        'and_sell': int(by_composite['and_sell_threshold']),
        'or': int(by_composite['or_threshold'])
    }

    print(f"  最优参数: buy < {params_composite['buy']}, AND > {params_composite['and_sell']}, OR > {params_composite['or']}")
    print(f"  训练期: 收益 {by_composite['total_return']:.1f}% | 夏普 {by_composite['sharpe_ratio']:.2f} | 回撤 {by_composite['max_drawdown']:.1f}% | 交易 {int(by_composite['num_trades'])} | 胜率 {by_composite['win_rate']:.0f}%")
    print(f"  综合评分: {by_composite['composite_score']:.4f}")

    # 检查参数是否相同
    params_same = (params_sharpe == params_composite)
    if params_same:
        print("\n✅ 两种方法选出相同参数!")
    else:
        print("\n⚠️ 两种方法选出不同参数，需要对比测试期表现")

    # ========================================
    # 阶段2: 测试期验证对比
    # ========================================
    print("\n" + "=" * 70)
    print("阶段2: 测试期验证对比 (2025)")
    print("=" * 70)

    result_sharpe = test_with_params(train_price, train_sentiment, test_price, test_sentiment,
                                      params_sharpe, "Sharpe-Only")
    result_composite = test_with_params(train_price, train_sentiment, test_price, test_sentiment,
                                         params_composite, "Composite")

    # 打印对比结果
    print(f"\n{'='*90}")
    print(f"{'指标':<20} {'夏普比率排序':>25} {'综合评分排序':>25} {'差异':>15}")
    print(f"{'='*90}")

    # 参数
    print(f"{'--- 参数 ---':<20}")
    print(f"{'  buy_threshold':<20} {params_sharpe['buy']:>25} {params_composite['buy']:>25}")
    print(f"{'  and_sell_threshold':<20} {params_sharpe['and_sell']:>25} {params_composite['and_sell']:>25}")
    print(f"{'  or_threshold':<20} {params_sharpe['or']:>25} {params_composite['or']:>25}")

    # 训练期
    print(f"\n{'--- 训练期 (2021-2024) ---':<20}")
    s_train = result_sharpe['train']['metrics']
    c_train = result_composite['train']['metrics']
    print(f"{'  总收益':<20} {s_train['total_return']*100:>24.2f}% {c_train['total_return']*100:>24.2f}%")
    print(f"{'  夏普比率':<20} {s_train['sharpe_ratio']:>25.2f} {c_train['sharpe_ratio']:>25.2f}")
    print(f"{'  最大回撤':<20} {s_train['max_drawdown']*100:>24.2f}% {c_train['max_drawdown']*100:>24.2f}%")
    print(f"{'  交易次数':<20} {len(result_sharpe['train']['trades']):>25} {len(result_composite['train']['trades']):>25}")
    print(f"{'  胜率':<20} {s_train.get('trade_win_rate',0)*100:>24.1f}% {c_train.get('trade_win_rate',0)*100:>24.1f}%")

    # 测试期
    print(f"\n{'--- 测试期 (2025) ---':<20}")
    s_test = result_sharpe['test']['metrics']
    c_test = result_composite['test']['metrics']
    s_ret = s_test['total_return']*100
    c_ret = c_test['total_return']*100
    s_sharpe = s_test['sharpe_ratio']
    c_sharpe = c_test['sharpe_ratio']

    print(f"{'  总收益':<20} {s_ret:>24.2f}% {c_ret:>24.2f}% {c_ret-s_ret:>+14.2f}%")
    print(f"{'  夏普比率':<20} {s_sharpe:>25.2f} {c_sharpe:>25.2f} {c_sharpe-s_sharpe:>+15.2f}")
    print(f"{'  最大回撤':<20} {s_test['max_drawdown']*100:>24.2f}% {c_test['max_drawdown']*100:>24.2f}%")
    print(f"{'  交易次数':<20} {len(result_sharpe['test']['trades']):>25} {len(result_composite['test']['trades']):>25}")
    print(f"{'  胜率':<20} {s_test.get('trade_win_rate',0)*100:>24.1f}% {c_test.get('trade_win_rate',0)*100:>24.1f}%")
    print(f"{'='*90}")

    # ========================================
    # 阶段3: 结论
    # ========================================
    print("\n" + "=" * 70)
    print("阶段3: 结论")
    print("=" * 70)

    # 判断哪个更好
    if c_ret > s_ret:
        winner = "综合评分"
        diff = c_ret - s_ret
    elif s_ret > c_ret:
        winner = "夏普比率"
        diff = s_ret - c_ret
    else:
        winner = "平局"
        diff = 0

    if not params_same:
        print(f"\n🏆 测试期收益更高: {winner} (+{diff:.2f}%)")

        if c_sharpe > s_sharpe:
            print(f"🏆 测试期夏普更高: 综合评分 (+{c_sharpe-s_sharpe:.2f})")
        elif s_sharpe > c_sharpe:
            print(f"🏆 测试期夏普更高: 夏普比率 (+{s_sharpe-c_sharpe:.2f})")
    else:
        print("\n两种方法参数相同，无需对比")

    # ========================================
    # 保存结果
    # ========================================
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.dirname(__file__)

    # 保存搜索结果（包含综合评分）
    train_results.to_csv(os.path.join(base_dir, f'train_grid_search_{SYMBOL}_{timestamp}.csv'), index=False)

    # 选择推荐的方法
    if c_ret >= s_ret:
        best_method = "Composite"
        best_params = params_composite
        best_result = result_composite
    else:
        best_method = "Sharpe"
        best_params = params_sharpe
        best_result = result_sharpe

    # 保存最优参数
    with open(os.path.join(base_dir, f'best_params_{SYMBOL}.txt'), 'w') as f:
        f.write(f"实验: {SYMBOL} Walk-Forward 网格搜索\n")
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"训练期: {TRAIN_START} ~ {TRAIN_END}\n")
        f.write(f"测试期: {TEST_START} ~ {TEST_END}\n\n")
        f.write(f"推荐方法: {best_method}\n\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"方法1: 夏普比率排序\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"参数: buy < {params_sharpe['buy']}, AND > {params_sharpe['and_sell']}, OR > {params_sharpe['or']}\n")
        f.write(f"训练期收益: {s_train['total_return']*100:.2f}%\n")
        f.write(f"测试期收益: {s_ret:.2f}%\n")
        f.write(f"测试期夏普: {s_sharpe:.2f}\n\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"方法2: 综合评分排序\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"参数: buy < {params_composite['buy']}, AND > {params_composite['and_sell']}, OR > {params_composite['or']}\n")
        f.write(f"训练期收益: {c_train['total_return']*100:.2f}%\n")
        f.write(f"测试期收益: {c_ret:.2f}%\n")
        f.write(f"测试期夏普: {c_sharpe:.2f}\n")

    # 生成两张可视化图
    viz_sharpe = visualize_comparison(
        result_sharpe['train'], result_sharpe['test'],
        SYMBOL, params_sharpe, "Sharpe"
    )

    viz_composite = visualize_comparison(
        result_composite['train'], result_composite['test'],
        SYMBOL, params_composite, "Composite"
    )

    print(f"\n" + "=" * 70)
    print("✅ Walk-Forward 对比验证完成!")
    print("=" * 70)
    print(f"\n已保存文件:")
    print(f"  - train_grid_search_{SYMBOL}_{timestamp}.csv")
    print(f"  - best_params_{SYMBOL}.txt")
    print(f"  - walk_forward_{SYMBOL}_Sharpe.png")
    print(f"  - walk_forward_{SYMBOL}_Composite.png")

    return result_sharpe, result_composite


if __name__ == "__main__":
    main()
