"""
纯情绪交易策略 - 不使用ML模型

策略逻辑:
  买入: smoothed_index < buy_threshold (恐惧时买入)
  卖出: smoothed_index > sell_threshold (贪婪时卖出)

Walk-Forward 验证:
  训练集 (4年): 网格搜索找候选参数
  验证集 (1年): 选择最优阈值组合
  测试集 (1年): 评估策略表现

Author: Claude
Date: 2026-01
"""
import sys
import os

# 添加父目录的 src 到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(parent_dir, 'src'))

import pandas as pd
import numpy as np
from datetime import datetime
import psycopg2
from itertools import product

from data_loader import DataLoader
from backtest_engine import EnhancedBacktester

# ============================================================
# 配置参数
# ============================================================

SYMBOL = "TSLA"

db_config = {
    "host": "localhost",
    "port": 5432,
    "database": "crypto_fear_greed_2",
    "user": "sc2025",
    "password": ""
}

# 回测参数 (纯情绪信号，无止盈止损)
BACKTEST_PARAMS = {
    "initial_capital": 100000,
    "commission_rate": 0.001,
    "slippage_rate": 0.001,
    "take_profit_pct": 999.0,   # 禁用止盈
    "stop_loss_pct": 999.0,     # 禁用止损
    "max_holding_days": 999,    # 禁用超时
    "position_pct": 0.8
}

# 网格搜索参数范围
GRID_SEARCH_PARAMS = {
    "buy_thresholds": list(range(-30, 5, 5)),    # -30, -25, -20, ..., 0
    "sell_thresholds": list(range(0, 35, 5)),    # 0, 5, 10, ..., 30
}

# Walk-Forward 周期配置
TRAIN_YEARS = 4
VAL_YEARS = 1
TEST_YEARS = 1

# 5个验证周期
VALIDATION_PERIODS = [
    {"year": 2021, "test_start": "2021-01-01", "test_end": "2021-12-31", "label": "2021 (复苏牛市)"},
    {"year": 2022, "test_start": "2022-01-01", "test_end": "2022-12-31", "label": "2022 (加息熊市)"},
    {"year": 2023, "test_start": "2023-01-01", "test_end": "2023-12-31", "label": "2023 (震荡反弹)"},
    {"year": 2024, "test_start": "2024-01-01", "test_end": "2024-12-31", "label": "2024 (AI牛市)"},
    {"year": 2025, "test_start": "2025-01-01", "test_end": "2025-12-31", "label": "2025 (科技牛市)"},
]


# ============================================================
# 数据加载
# ============================================================

def load_sentiment_data(db_config, symbol, start_date="2014-01-01"):
    """加载情绪指标数据"""
    conn = psycopg2.connect(**db_config)
    query = f"""
        SELECT date, smoothed_index
        FROM fear_greed_index
        WHERE symbol = '{symbol}' AND date >= '{start_date}'
        ORDER BY date
    """
    df = pd.read_sql(query, conn)
    conn.close()

    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.set_index('date')
    return df


# ============================================================
# 信号生成
# ============================================================

def generate_sentiment_signals(sentiment_series, buy_threshold, sell_threshold):
    """
    根据情绪阈值生成交易信号

    Parameters:
    -----------
    sentiment_series : pd.Series
        情绪指标序列 (smoothed_index)
    buy_threshold : float
        买入阈值 (情绪 < 此值时买入)
    sell_threshold : float
        卖出阈值 (情绪 > 此值时卖出)

    Returns:
    --------
    pd.DataFrame
        包含 buy_signal, sell_signal 的信号 DataFrame
    """
    signals = pd.DataFrame(index=sentiment_series.index)
    signals['smoothed_index'] = sentiment_series.values
    signals['buy_signal'] = (sentiment_series < buy_threshold).astype(int)
    signals['sell_signal'] = (sentiment_series > sell_threshold).astype(int)
    signals['position_size'] = 0

    return signals


# ============================================================
# 回测执行
# ============================================================

def run_backtest(signals, price_data, params):
    """运行回测并返回指标"""
    backtester = EnhancedBacktester(
        initial_capital=params["initial_capital"],
        commission_rate=params["commission_rate"],
        slippage_rate=params["slippage_rate"],
        take_profit_pct=params["take_profit_pct"],
        stop_loss_pct=params["stop_loss_pct"],
        max_holding_days=params["max_holding_days"],
        use_dynamic_position=True,
        position_pct=params["position_pct"]
    )

    portfolio, metrics, trades = backtester.run_backtest_with_sell_signal(signals, price_data)

    return metrics, trades


def calculate_composite_score(metrics):
    """计算综合评分"""
    total_return = metrics.get('total_return', 0) or 0
    sharpe = metrics.get('sharpe_ratio', 0) or 0
    max_dd = abs(metrics.get('max_drawdown', 0) or 0)

    # 综合评分: 0.4*收益 + 0.4*(夏普/3) - 0.2*回撤
    composite = 0.4 * total_return + 0.4 * (sharpe / 3.0) - 0.2 * max_dd
    return composite


# ============================================================
# 网格搜索
# ============================================================

def grid_search_thresholds(sentiment_data, price_data, date_start, date_end,
                           buy_thresholds, sell_thresholds, backtest_params):
    """
    在给定数据范围内进行网格搜索

    Returns:
    --------
    list: 所有参数组合的结果 [(buy_th, sell_th, metrics, score), ...]
    """
    results = []

    # 筛选日期范围
    mask = (sentiment_data.index >= date_start) & (sentiment_data.index <= date_end)
    sentiment_period = sentiment_data[mask]['smoothed_index']

    price_mask = (price_data.index >= date_start) & (price_data.index <= date_end)
    price_period = price_data[price_mask]

    if len(sentiment_period) == 0 or len(price_period) == 0:
        return results

    # 对齐数据
    common_dates = sentiment_period.index.intersection(price_period.index)
    sentiment_period = sentiment_period.reindex(common_dates)
    price_period = price_period.reindex(common_dates)

    # 网格搜索
    for buy_th, sell_th in product(buy_thresholds, sell_thresholds):
        # 买入阈值必须小于卖出阈值 (有意义的策略)
        if buy_th >= sell_th:
            continue

        # 生成信号
        signals = generate_sentiment_signals(sentiment_period, buy_th, sell_th)

        # 检查是否有足够的信号
        if signals['buy_signal'].sum() == 0:
            continue

        # 运行回测
        try:
            metrics, trades = run_backtest(signals, price_period, backtest_params)
            score = calculate_composite_score(metrics)

            results.append({
                'buy_threshold': buy_th,
                'sell_threshold': sell_th,
                'total_return': metrics.get('total_return', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'total_trades': metrics.get('total_trades', 0),
                'win_rate': metrics.get('trade_win_rate', 0),
                'composite_score': score
            })
        except Exception as e:
            continue

    return results


# ============================================================
# Walk-Forward 验证
# ============================================================

def run_walk_forward_period(sentiment_data, price_data, period, backtest_params, grid_params):
    """
    运行单个 Walk-Forward 周期

    流程:
    1. 训练集 (4年): 网格搜索
    2. 验证集 (1年): 选择最优参数
    3. 测试集 (1年): 最终评估
    """
    test_year = period["year"]
    test_start = pd.to_datetime(period["test_start"], utc=True)
    test_end = pd.to_datetime(period["test_end"], utc=True)

    # 计算验证集和训练集日期
    val_start = test_start - pd.DateOffset(years=1)
    val_end = test_start - pd.Timedelta(days=1)

    train_start = val_start - pd.DateOffset(years=TRAIN_YEARS)
    train_end = val_start - pd.Timedelta(days=1)

    print(f"\n{'='*70}")
    print(f"周期: {period['label']}")
    print(f"{'='*70}")
    print(f"  训练集: {train_start.strftime('%Y-%m-%d')} ~ {train_end.strftime('%Y-%m-%d')}")
    print(f"  验证集: {val_start.strftime('%Y-%m-%d')} ~ {val_end.strftime('%Y-%m-%d')}")
    print(f"  测试集: {test_start.strftime('%Y-%m-%d')} ~ {test_end.strftime('%Y-%m-%d')}")

    # Step 1: 在训练集上网格搜索
    print(f"\n[1/3] 训练集网格搜索...")
    train_results = grid_search_thresholds(
        sentiment_data, price_data,
        train_start, train_end,
        grid_params["buy_thresholds"],
        grid_params["sell_thresholds"],
        backtest_params
    )

    if not train_results:
        print("  ❌ 训练集无有效结果")
        return None

    # 按综合评分排序，取 Top 5 候选
    train_results_sorted = sorted(train_results, key=lambda x: x['composite_score'], reverse=True)
    top_candidates = train_results_sorted[:5]

    print(f"  找到 {len(train_results)} 个有效组合")
    print(f"  Top 5 候选参数:")
    for i, r in enumerate(top_candidates):
        print(f"    {i+1}. buy<{r['buy_threshold']}, sell>{r['sell_threshold']}: "
              f"收益={r['total_return']:.2%}, 夏普={r['sharpe_ratio']:.2f}, 综合={r['composite_score']:.4f}")

    # Step 2: 在验证集上评估 Top 5，选择最优
    print(f"\n[2/3] 验证集选择最优参数...")
    best_val_score = -np.inf
    best_params = None
    best_val_metrics = None

    for candidate in top_candidates:
        buy_th = candidate['buy_threshold']
        sell_th = candidate['sell_threshold']

        # 在验证集上测试
        val_results = grid_search_thresholds(
            sentiment_data, price_data,
            val_start, val_end,
            [buy_th], [sell_th],
            backtest_params
        )

        if val_results:
            val_score = val_results[0]['composite_score']
            if val_score > best_val_score:
                best_val_score = val_score
                best_params = (buy_th, sell_th)
                best_val_metrics = val_results[0]

    if best_params is None:
        print("  ❌ 验证集无有效结果")
        return None

    print(f"  最优参数: buy<{best_params[0]}, sell>{best_params[1]}")
    print(f"  验证集表现: 收益={best_val_metrics['total_return']:.2%}, "
          f"夏普={best_val_metrics['sharpe_ratio']:.2f}")

    # Step 3: 在测试集上最终评估
    print(f"\n[3/3] 测试集最终评估...")
    test_results = grid_search_thresholds(
        sentiment_data, price_data,
        test_start, test_end,
        [best_params[0]], [best_params[1]],
        backtest_params
    )

    if not test_results:
        print("  ❌ 测试集无有效结果")
        return None

    test_metrics = test_results[0]

    print(f"\n  📊 测试集结果:")
    print(f"     收益率: {test_metrics['total_return']:.2%}")
    print(f"     夏普比率: {test_metrics['sharpe_ratio']:.2f}")
    print(f"     最大回撤: {test_metrics['max_drawdown']:.2%}")
    print(f"     交易次数: {test_metrics['total_trades']}")
    print(f"     胜率: {test_metrics['win_rate']:.2%}")
    print(f"     综合评分: {test_metrics['composite_score']:.4f}")

    # 情绪分布分析
    test_mask = (sentiment_data.index >= test_start) & (sentiment_data.index <= test_end)
    test_sentiment = sentiment_data[test_mask]['smoothed_index']
    buy_days = (test_sentiment < best_params[0]).sum()
    sell_days = (test_sentiment > best_params[1]).sum()

    print(f"\n  📈 信号分布:")
    print(f"     买入信号天数: {buy_days} ({buy_days/len(test_sentiment)*100:.1f}%)")
    print(f"     卖出信号天数: {sell_days} ({sell_days/len(test_sentiment)*100:.1f}%)")
    print(f"     平均情绪值: {test_sentiment.mean():.2f}")

    return {
        "year": test_year,
        "label": period["label"],
        "buy_threshold": best_params[0],
        "sell_threshold": best_params[1],
        "val_return": best_val_metrics['total_return'],
        "val_sharpe": best_val_metrics['sharpe_ratio'],
        "total_return": test_metrics['total_return'],
        "sharpe_ratio": test_metrics['sharpe_ratio'],
        "max_drawdown": test_metrics['max_drawdown'],
        "total_trades": test_metrics['total_trades'],
        "win_rate": test_metrics['win_rate'],
        "composite_score": test_metrics['composite_score'],
        "buy_signal_days": buy_days,
        "sell_signal_days": sell_days,
        "avg_sentiment": test_sentiment.mean()
    }


# ============================================================
# 主函数
# ============================================================

def main():
    print("="*70)
    print(f"纯情绪交易策略 Walk-Forward 验证 → {SYMBOL}")
    print("="*70)

    print("\n策略设计:")
    print("  买入条件: smoothed_index < buy_threshold (恐惧时买入)")
    print("  卖出条件: smoothed_index > sell_threshold (贪婪时卖出)")
    print(f"\n网格搜索范围:")
    print(f"  buy_threshold: {GRID_SEARCH_PARAMS['buy_thresholds']}")
    print(f"  sell_threshold: {GRID_SEARCH_PARAMS['sell_thresholds']}")

    print(f"\nWalk-Forward 配置:")
    print(f"  训练集: {TRAIN_YEARS}年")
    print(f"  验证集: {VAL_YEARS}年")
    print(f"  测试集: {TEST_YEARS}年")

    # 加载数据
    print("\n" + "="*70)
    print("加载数据...")
    print("="*70)

    loader = DataLoader(db_config)
    price_data = loader.load_ohlcv(SYMBOL, start_date="2014-01-01")
    loader.close()

    sentiment_data = load_sentiment_data(db_config, SYMBOL, start_date="2014-01-01")

    print(f"  价格数据: {len(price_data)} 条 ({price_data.index.min().strftime('%Y-%m-%d')} ~ {price_data.index.max().strftime('%Y-%m-%d')})")
    print(f"  情绪数据: {len(sentiment_data)} 条")
    print(f"  情绪统计: 均值={sentiment_data['smoothed_index'].mean():.2f}, "
          f"标准差={sentiment_data['smoothed_index'].std():.2f}")

    # 运行 Walk-Forward
    results = []
    for period in VALIDATION_PERIODS:
        result = run_walk_forward_period(
            sentiment_data, price_data, period,
            BACKTEST_PARAMS, GRID_SEARCH_PARAMS
        )
        if result:
            results.append(result)

    # 汇总结果
    print("\n" + "="*70)
    print("汇总结果")
    print("="*70)

    if results:
        print(f"\n{'周期':<18} {'阈值':<12} {'验证收益':>10} {'测试收益':>10} {'夏普':>8} {'回撤':>10} {'交易':>6} {'综合分':>10}")
        print("-"*95)

        for r in results:
            threshold_str = f"<{r['buy_threshold']},>{r['sell_threshold']}"
            print(f"{r['label']:<18} {threshold_str:<12} {r['val_return']:>10.2%} {r['total_return']:>10.2%} "
                  f"{r['sharpe_ratio']:>8.2f} {r['max_drawdown']:>10.2%} {r['total_trades']:>6} "
                  f"{r['composite_score']:>10.4f}")

        print("-"*95)

        # 平均值
        avg_return = np.mean([r['total_return'] for r in results])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
        avg_dd = np.mean([r['max_drawdown'] for r in results])
        avg_trades = np.mean([r['total_trades'] for r in results])
        avg_wr = np.mean([r['win_rate'] for r in results])
        avg_composite = np.mean([r['composite_score'] for r in results])

        print(f"{'平均':<18} {'':<12} {'':<10} {avg_return:>10.2%} "
              f"{avg_sharpe:>8.2f} {avg_dd:>10.2%} {avg_trades:>6.0f} "
              f"{avg_composite:>10.4f}")

        # 稳定性分析
        print("\n稳定性分析:")
        positive_years = sum(1 for r in results if r['total_return'] > 0)
        print(f"  盈利年份: {positive_years}/{len(results)}")
        print(f"  收益率标准差: {np.std([r['total_return'] for r in results]):.2%}")
        print(f"  夏普率标准差: {np.std([r['sharpe_ratio'] for r in results]):.2f}")

        # 阈值分析
        print("\n阈值选择分析:")
        print(f"  {'周期':<18} {'买入阈值':>10} {'卖出阈值':>10} {'阈值差':>10}")
        print("  " + "-"*50)
        for r in results:
            diff = r['sell_threshold'] - r['buy_threshold']
            print(f"  {r['label']:<18} {r['buy_threshold']:>10} {r['sell_threshold']:>10} {diff:>10}")

        # 熊市表现
        bear_result = next((r for r in results if r['year'] == 2022), None)
        if bear_result:
            print(f"\n熊市(2022)表现:")
            print(f"  最优阈值: buy<{bear_result['buy_threshold']}, sell>{bear_result['sell_threshold']}")
            print(f"  收益率: {bear_result['total_return']:.2%}")
            print(f"  最大回撤: {bear_result['max_drawdown']:.2%}")
            print(f"  平均情绪: {bear_result['avg_sentiment']:.2f}")

    print("\n" + "="*70)
    print("实验完成!")
    print("="*70)

    return results


if __name__ == "__main__":
    results = main()
