"""
详细分析：显示每个周期的训练、验证、测试期阈值和交易明细
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import psycopg2
from itertools import product

from data_loader import DataLoader
from backtest_engine import EnhancedBacktester

# 配置
SYMBOL = "TSLA"
db_config = {
    "host": "localhost",
    "port": 5432,
    "database": "crypto_fear_greed_2",
    "user": "sc2025",
    "password": ""
}

# 回测参数 (无止盈止损)
BACKTEST_PARAMS = {
    "initial_capital": 100000,
    "commission_rate": 0.001,
    "slippage_rate": 0.001,
    "take_profit_pct": 999.0,
    "stop_loss_pct": 999.0,
    "max_holding_days": 999,
    "position_pct": 0.8
}

# 网格搜索参数
GRID_SEARCH_PARAMS = {
    "buy_thresholds": list(range(-30, 5, 5)),
    "sell_thresholds": list(range(0, 35, 5)),
}

# 验证周期
VALIDATION_PERIODS = [
    {"year": 2021, "label": "2021 (复苏牛市)"},
    {"year": 2022, "label": "2022 (加息熊市)"},
    {"year": 2023, "label": "2023 (震荡反弹)"},
    {"year": 2024, "label": "2024 (AI牛市)"},
    {"year": 2025, "label": "2025 (科技牛市)"},
]


def load_sentiment_data(db_config, symbol):
    """加载情绪数据"""
    conn = psycopg2.connect(**db_config)
    query = f"""
        SELECT date, smoothed_index
        FROM fear_greed_index
        WHERE symbol = '{symbol}'
        ORDER BY date
    """
    df = pd.read_sql(query, conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.set_index('date')
    return df


def run_backtest(stock_data, sentiment_data, buy_threshold, sell_threshold, start_date, end_date):
    """运行单次回测"""
    mask = (stock_data.index >= start_date) & (stock_data.index <= end_date)
    price_data = stock_data[mask]
    sent_data = sentiment_data.reindex(price_data.index)

    if len(price_data) == 0:
        return None, None, None

    signals = pd.DataFrame(index=price_data.index)
    signals['smoothed_index'] = sent_data['smoothed_index']
    signals['buy_signal'] = (signals['smoothed_index'] < buy_threshold).astype(int)
    signals['sell_signal'] = (signals['smoothed_index'] > sell_threshold).astype(int)
    signals['prob_profit'] = 0.5
    signals['position_size'] = 0

    backtester = EnhancedBacktester(
        initial_capital=BACKTEST_PARAMS["initial_capital"],
        commission_rate=BACKTEST_PARAMS["commission_rate"],
        slippage_rate=BACKTEST_PARAMS["slippage_rate"],
        take_profit_pct=BACKTEST_PARAMS["take_profit_pct"],
        stop_loss_pct=BACKTEST_PARAMS["stop_loss_pct"],
        max_holding_days=BACKTEST_PARAMS["max_holding_days"],
        use_dynamic_position=True,
        position_pct=BACKTEST_PARAMS["position_pct"]
    )

    portfolio, metrics, trades = backtester.run_backtest_with_sell_signal(signals, price_data)
    return portfolio, metrics, trades


def grid_search(stock_data, sentiment_data, start_date, end_date):
    """网格搜索最优参数"""
    results = []
    for buy_th, sell_th in product(GRID_SEARCH_PARAMS["buy_thresholds"],
                                    GRID_SEARCH_PARAMS["sell_thresholds"]):
        if buy_th >= sell_th:
            continue

        portfolio, metrics, trades = run_backtest(
            stock_data, sentiment_data, buy_th, sell_th, start_date, end_date
        )

        if metrics and metrics.get('total_trades', 0) > 0:
            total_return = metrics.get('total_return', 0) or 0
            sharpe = metrics.get('sharpe_ratio', 0) or 0
            max_dd = abs(metrics.get('max_drawdown', 0) or 0)
            composite = 0.4 * total_return + 0.4 * (sharpe / 3.0) - 0.2 * max_dd

            results.append({
                'buy_threshold': buy_th,
                'sell_threshold': sell_th,
                'total_return': total_return,
                'sharpe_ratio': sharpe,
                'max_drawdown': -max_dd,
                'composite_score': composite,
                'trades': trades
            })

    if results:
        results.sort(key=lambda x: x['composite_score'], reverse=True)
    return results


def analyze_period(stock_data, sentiment_data, period):
    """分析单个周期"""
    year = period["year"]

    # 计算各期时间范围 (3年训练)
    train_start = f"{year - 4}-01-01"
    train_end = f"{year - 2}-12-31"
    val_start = f"{year - 1}-01-01"
    val_end = f"{year - 1}-12-31"
    test_start = f"{year}-01-01"
    test_end = f"{year}-12-31"

    print(f"\n{'='*80}")
    print(f"周期: {period['label']}")
    print(f"{'='*80}")
    print(f"  训练期: {train_start} ~ {train_end}")
    print(f"  验证期: {val_start} ~ {val_end}")
    print(f"  测试期: {test_start} ~ {test_end}")

    # 1. 训练期网格搜索
    print(f"\n[训练期] 网格搜索...")
    train_results = grid_search(stock_data, sentiment_data, train_start, train_end)

    if not train_results:
        print("  ❌ 无有效结果")
        return None

    top5 = train_results[:5]
    print(f"  Top 5 候选参数:")
    for i, r in enumerate(top5, 1):
        print(f"    {i}. buy<{r['buy_threshold']}, sell>{r['sell_threshold']}: "
              f"收益={r['total_return']:.2%}, 夏普={r['sharpe_ratio']:.2f}")

    # 2. 验证期选择最优
    print(f"\n[验证期] 评估候选参数...")
    best_val_result = None
    best_val_score = -999

    for candidate in top5:
        portfolio, metrics, trades = run_backtest(
            stock_data, sentiment_data,
            candidate['buy_threshold'], candidate['sell_threshold'],
            val_start, val_end
        )

        if metrics and metrics.get('total_trades', 0) > 0:
            total_return = metrics.get('total_return', 0) or 0
            sharpe = metrics.get('sharpe_ratio', 0) or 0
            max_dd = abs(metrics.get('max_drawdown', 0) or 0)
            score = 0.4 * total_return + 0.4 * (sharpe / 3.0) - 0.2 * max_dd

            print(f"    buy<{candidate['buy_threshold']}, sell>{candidate['sell_threshold']}: "
                  f"收益={total_return:.2%}, 夏普={sharpe:.2f}")

            if score > best_val_score:
                best_val_score = score
                best_val_result = {
                    'buy_threshold': candidate['buy_threshold'],
                    'sell_threshold': candidate['sell_threshold'],
                    'val_return': total_return,
                    'val_sharpe': sharpe,
                    'val_trades': trades
                }

    if not best_val_result:
        print("  ❌ 验证期无有效结果")
        return None

    print(f"\n  ✓ 最优参数: buy<{best_val_result['buy_threshold']}, sell>{best_val_result['sell_threshold']}")
    print(f"    验证期收益: {best_val_result['val_return']:.2%}")

    # 打印验证期交易明细
    if best_val_result['val_trades']:
        print(f"\n  [验证期交易明细]")
        for t in best_val_result['val_trades']:
            entry = str(t['entry_date'])[:10]
            exit_d = str(t.get('exit_date', ''))[:10] if pd.notna(t.get('exit_date')) else 'Open'
            ret = t.get('return_pct', 0)
            ret_str = f"{ret:.2%}" if ret else '-'
            print(f"    买入: {entry} @ ${t['entry_price']:.2f} → 卖出: {exit_d} @ ${t.get('exit_price', 0):.2f} | 收益: {ret_str}")

    # 3. 测试期最终评估
    print(f"\n[测试期] 最终评估...")
    portfolio, metrics, trades = run_backtest(
        stock_data, sentiment_data,
        best_val_result['buy_threshold'], best_val_result['sell_threshold'],
        test_start, test_end
    )

    if not metrics:
        print("  ❌ 测试期无结果")
        return None

    test_return = metrics.get('total_return', 0) or 0
    test_sharpe = metrics.get('sharpe_ratio', 0) or 0
    test_dd = metrics.get('max_drawdown', 0) or 0

    print(f"  测试期收益: {test_return:.2%}")
    print(f"  测试期夏普: {test_sharpe:.2f}")
    print(f"  测试期回撤: {test_dd:.2%}")

    # 打印测试期交易明细
    if trades:
        print(f"\n  [测试期交易明细]")
        for t in trades:
            entry = str(t['entry_date'])[:10]
            exit_d = str(t.get('exit_date', ''))[:10] if pd.notna(t.get('exit_date')) else 'Open'
            ret = t.get('return_pct', 0)
            ret_str = f"{ret:.2%}" if ret else '-'
            reason = t.get('exit_reason', '-')
            print(f"    买入: {entry} @ ${t['entry_price']:.2f} → 卖出: {exit_d} @ ${t.get('exit_price', 0):.2f} | 收益: {ret_str} | {reason}")

    return {
        'year': year,
        'label': period['label'],
        'train_period': f"{train_start} ~ {train_end}",
        'val_period': f"{val_start} ~ {val_end}",
        'test_period': f"{test_start} ~ {test_end}",
        'buy_threshold': best_val_result['buy_threshold'],
        'sell_threshold': best_val_result['sell_threshold'],
        'val_return': best_val_result['val_return'],
        'test_return': test_return,
        'test_sharpe': test_sharpe,
        'test_drawdown': test_dd,
        'test_trades': trades
    }


def main():
    print("=" * 80)
    print("TSLA 纯情绪策略 - 详细分析")
    print("=" * 80)

    # 加载数据
    print("\n加载数据...")
    loader = DataLoader(db_config)
    stock_data = loader.load_ohlcv(SYMBOL, start_date="2014-01-01")
    loader.close()
    sentiment_data = load_sentiment_data(db_config, SYMBOL)
    print(f"  价格数据: {len(stock_data)} 条")
    print(f"  情绪数据: {len(sentiment_data)} 条")

    # 分析每个周期
    results = []
    for period in VALIDATION_PERIODS:
        result = analyze_period(stock_data, sentiment_data, period)
        if result:
            results.append(result)

    # 汇总
    print("\n" + "=" * 80)
    print("汇总表格")
    print("=" * 80)

    print(f"\n{'周期':<18} {'训练期':<25} {'验证期':<25} {'测试期':<25}")
    print("-" * 95)
    for r in results:
        print(f"{r['label']:<18} {r['train_period']:<25} {r['val_period']:<25} {r['test_period']:<25}")

    print(f"\n{'周期':<18} {'阈值':<12} {'验证收益':>10} {'测试收益':>10} {'夏普':>8} {'回撤':>10}")
    print("-" * 75)
    for r in results:
        threshold = f"<{r['buy_threshold']},>{r['sell_threshold']}"
        print(f"{r['label']:<18} {threshold:<12} {r['val_return']:>10.2%} {r['test_return']:>10.2%} "
              f"{r['test_sharpe']:>8.2f} {r['test_drawdown']:>10.2%}")

    if results:
        avg_return = np.mean([r['test_return'] for r in results])
        avg_sharpe = np.mean([r['test_sharpe'] for r in results])
        avg_dd = np.mean([r['test_drawdown'] for r in results])
        print("-" * 75)
        print(f"{'平均':<18} {'':<12} {'':<10} {avg_return:>10.2%} {avg_sharpe:>8.2f} {avg_dd:>10.2%}")

    print("\n" + "=" * 80)
    print("完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
