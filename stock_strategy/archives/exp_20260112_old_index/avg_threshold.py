"""
平均阈值策略：
1. 对训练期每年单独找最优阈值
2. 对4年的阈值取平均
3. 用平均阈值测试第5年
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

# 测试周期 (4年训练 + 1年测试)
TEST_PERIODS = [
    {"test_year": 2021, "train_years": [2017, 2018, 2019, 2020]},
    {"test_year": 2022, "train_years": [2018, 2019, 2020, 2021]},
    {"test_year": 2023, "train_years": [2019, 2020, 2021, 2022]},
    {"test_year": 2024, "train_years": [2020, 2021, 2022, 2023]},
    {"test_year": 2025, "train_years": [2021, 2022, 2023, 2024]},
]


def load_sentiment_data(db_config, symbol):
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


def find_best_threshold_for_year(stock_data, sentiment_data, year):
    """找单年最优阈值"""
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    best_result = None
    best_score = -999

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
            # 综合评分
            score = 0.4 * total_return + 0.4 * (sharpe / 3.0) - 0.2 * max_dd

            if score > best_score:
                best_score = score
                best_result = {
                    'buy_threshold': buy_th,
                    'sell_threshold': sell_th,
                    'return': total_return,
                    'sharpe': sharpe,
                    'score': score
                }

    return best_result


def analyze_period(stock_data, sentiment_data, period):
    """分析单个测试周期"""
    test_year = period["test_year"]
    train_years = period["train_years"]

    print(f"\n{'='*70}")
    print(f"测试年份: {test_year}")
    print(f"训练年份: {train_years}")
    print(f"{'='*70}")

    # 1. 对每个训练年份找最优阈值
    print(f"\n[Step 1] 各年最优阈值:")
    yearly_best = []

    for year in train_years:
        best = find_best_threshold_for_year(stock_data, sentiment_data, year)
        if best:
            yearly_best.append(best)
            print(f"  {year}: buy<{best['buy_threshold']}, sell>{best['sell_threshold']} "
                  f"| 收益={best['return']:.2%}, 夏普={best['sharpe']:.2f}")
        else:
            print(f"  {year}: 无有效结果")

    if len(yearly_best) == 0:
        print("  ❌ 无法计算平均阈值")
        return None

    # 2. 计算平均阈值
    avg_buy = np.mean([r['buy_threshold'] for r in yearly_best])
    avg_sell = np.mean([r['sell_threshold'] for r in yearly_best])

    # 四舍五入到最近的5的倍数
    avg_buy_rounded = round(avg_buy / 5) * 5
    avg_sell_rounded = round(avg_sell / 5) * 5

    print(f"\n[Step 2] 平均阈值:")
    print(f"  原始平均: buy<{avg_buy:.1f}, sell>{avg_sell:.1f}")
    print(f"  四舍五入: buy<{avg_buy_rounded}, sell>{avg_sell_rounded}")

    # 3. 用平均阈值测试
    print(f"\n[Step 3] 测试 {test_year} 年:")
    test_start = f"{test_year}-01-01"
    test_end = f"{test_year}-12-31"

    portfolio, metrics, trades = run_backtest(
        stock_data, sentiment_data,
        avg_buy_rounded, avg_sell_rounded,
        test_start, test_end
    )

    if not metrics:
        print("  ❌ 测试无结果")
        return None

    test_return = metrics.get('total_return', 0) or 0
    test_sharpe = metrics.get('sharpe_ratio', 0) or 0
    test_dd = metrics.get('max_drawdown', 0) or 0
    n_trades = metrics.get('total_trades', 0)

    print(f"  收益率: {test_return:.2%}")
    print(f"  夏普比率: {test_sharpe:.2f}")
    print(f"  最大回撤: {test_dd:.2%}")
    print(f"  交易次数: {n_trades}")

    if trades:
        print(f"\n  [交易明细]")
        for t in trades:
            entry = str(t['entry_date'])[:10]
            exit_d = str(t.get('exit_date', ''))[:10] if pd.notna(t.get('exit_date')) else 'Open'
            print(f"    {entry} @${t['entry_price']:.0f} → {exit_d} @${t.get('exit_price', 0):.0f}")

    return {
        'test_year': test_year,
        'train_years': train_years,
        'yearly_thresholds': yearly_best,
        'avg_buy': avg_buy_rounded,
        'avg_sell': avg_sell_rounded,
        'test_return': test_return,
        'test_sharpe': test_sharpe,
        'test_drawdown': test_dd,
        'n_trades': n_trades
    }


def main():
    print("=" * 70)
    print("TSLA 平均阈值策略测试")
    print("=" * 70)
    print("\n方法: 4年各自找最优阈值 → 取平均 → 测试第5年")

    # 加载数据
    print("\n加载数据...")
    loader = DataLoader(db_config)
    stock_data = loader.load_ohlcv(SYMBOL, start_date="2014-01-01")
    loader.close()
    sentiment_data = load_sentiment_data(db_config, SYMBOL)
    print(f"  价格数据: {len(stock_data)} 条")
    print(f"  情绪数据: {len(sentiment_data)} 条")

    # 测试每个周期
    results = []
    for period in TEST_PERIODS:
        result = analyze_period(stock_data, sentiment_data, period)
        if result:
            results.append(result)

    # 汇总
    print("\n" + "=" * 70)
    print("汇总结果")
    print("=" * 70)

    print(f"\n{'测试年':<8} {'平均阈值':<15} {'测试收益':>10} {'夏普':>8} {'回撤':>10} {'交易':>6}")
    print("-" * 65)

    for r in results:
        threshold = f"<{r['avg_buy']},>{r['avg_sell']}"
        print(f"{r['test_year']:<8} {threshold:<15} {r['test_return']:>10.2%} "
              f"{r['test_sharpe']:>8.2f} {r['test_drawdown']:>10.2%} {r['n_trades']:>6}")

    if results:
        avg_return = np.mean([r['test_return'] for r in results])
        avg_sharpe = np.mean([r['test_sharpe'] for r in results])
        avg_dd = np.mean([r['test_drawdown'] for r in results])
        print("-" * 65)
        print(f"{'平均':<8} {'':<15} {avg_return:>10.2%} {avg_sharpe:>8.2f} {avg_dd:>10.2%}")

        # 累计收益
        cumulative = 1.0
        for r in results:
            cumulative *= (1 + r['test_return'])
        print(f"\n5年累计收益: {(cumulative-1):.2%}")
        print(f"年化收益: {(cumulative**(1/5)-1):.2%}")

        # 盈利年份
        profitable = sum(1 for r in results if r['test_return'] > 0)
        print(f"盈利年份: {profitable}/{len(results)}")

    # 对比固定阈值
    print("\n" + "=" * 70)
    print("与固定阈值对比")
    print("=" * 70)
    print(f"\n{'方法':<20} {'平均收益':>10} {'夏普':>8} {'5年累计':>12} {'年化':>10}")
    print("-" * 65)
    print(f"{'固定 <0, >30':<20} {'48.30%':>10} {'0.94':>8} {'299.75%':>12} {'31.93%':>10}")
    print(f"{'固定 <-10, >30':<20} {'46.04%':>10} {'1.06':>8} {'336.37%':>12} {'34.27%':>10}")
    if results:
        print(f"{'平均阈值法':<20} {avg_return:>10.2%} {avg_sharpe:>8.2f} {(cumulative-1):>12.2%} {(cumulative**(1/5)-1):>10.2%}")

    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
