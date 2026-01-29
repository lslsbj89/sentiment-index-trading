"""
七姐妹平均阈值策略测试
方法: 4年各自找最优阈值 → 取平均 → 测试第5年
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

# 七姐妹股票
MAG7_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

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

# 测试周期
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
    if len(df) == 0:
        return None
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.set_index('date')
    return df


def run_backtest(stock_data, sentiment_data, buy_threshold, sell_threshold, start_date, end_date):
    mask = (stock_data.index >= start_date) & (stock_data.index <= end_date)
    price_data = stock_data[mask]

    if sentiment_data is None:
        return None, None, None

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


def test_symbol_avg_threshold(symbol, loader):
    """测试单个股票的平均阈值策略"""
    print(f"\n{'='*60}")
    print(f"股票: {symbol}")
    print(f"{'='*60}")

    # 加载数据
    try:
        stock_data = loader.load_ohlcv(symbol, start_date="2014-01-01")
        if len(stock_data) == 0:
            print(f"  ❌ 无价格数据")
            return None
    except:
        print(f"  ❌ 加载价格失败")
        return None

    sentiment_data = load_sentiment_data(db_config, symbol)
    if sentiment_data is None:
        print(f"  ❌ 无情绪数据")
        return None

    # 测试每个周期
    results = []

    for period in TEST_PERIODS:
        test_year = period["test_year"]
        train_years = period["train_years"]

        # 找各年最优阈值
        yearly_best = []
        for year in train_years:
            best = find_best_threshold_for_year(stock_data, sentiment_data, year)
            if best:
                yearly_best.append(best)

        if len(yearly_best) == 0:
            results.append({
                'test_year': test_year,
                'avg_buy': None,
                'avg_sell': None,
                'return': 0,
                'sharpe': 0,
                'drawdown': 0
            })
            continue

        # 计算平均阈值
        avg_buy = np.mean([r['buy_threshold'] for r in yearly_best])
        avg_sell = np.mean([r['sell_threshold'] for r in yearly_best])
        avg_buy_rounded = round(avg_buy / 5) * 5
        avg_sell_rounded = round(avg_sell / 5) * 5

        # 测试
        test_start = f"{test_year}-01-01"
        test_end = f"{test_year}-12-31"

        portfolio, metrics, trades = run_backtest(
            stock_data, sentiment_data,
            avg_buy_rounded, avg_sell_rounded,
            test_start, test_end
        )

        if metrics:
            results.append({
                'test_year': test_year,
                'avg_buy': avg_buy_rounded,
                'avg_sell': avg_sell_rounded,
                'return': metrics.get('total_return', 0) or 0,
                'sharpe': metrics.get('sharpe_ratio', 0) or 0,
                'drawdown': metrics.get('max_drawdown', 0) or 0
            })
        else:
            results.append({
                'test_year': test_year,
                'avg_buy': avg_buy_rounded,
                'avg_sell': avg_sell_rounded,
                'return': 0,
                'sharpe': 0,
                'drawdown': 0
            })

    # 打印年度结果
    print(f"\n  {'年份':<6} {'阈值':<12} {'收益率':>10} {'夏普':>8} {'回撤':>10}")
    print(f"  {'-'*50}")
    for r in results:
        if r['avg_buy'] is not None:
            threshold = f"<{r['avg_buy']},>{r['avg_sell']}"
        else:
            threshold = "N/A"
        print(f"  {r['test_year']:<6} {threshold:<12} {r['return']:>10.2%} {r['sharpe']:>8.2f} {r['drawdown']:>10.2%}")

    # 汇总
    valid_results = [r for r in results if r['return'] != 0 or r['avg_buy'] is not None]
    if valid_results:
        avg_return = np.mean([r['return'] for r in valid_results])
        avg_sharpe = np.mean([r['sharpe'] for r in valid_results])
        avg_dd = np.mean([r['drawdown'] for r in valid_results])
        profitable = sum(1 for r in valid_results if r['return'] > 0)

        cumulative = 1.0
        for r in valid_results:
            cumulative *= (1 + r['return'])

        print(f"  {'-'*50}")
        print(f"  {'平均':<6} {'':<12} {avg_return:>10.2%} {avg_sharpe:>8.2f} {avg_dd:>10.2%}")
        print(f"\n  盈利年份: {profitable}/{len(valid_results)}")
        print(f"  5年累计: {(cumulative-1):.2%}")
        print(f"  年化收益: {(cumulative**(1/len(valid_results))-1):.2%}")

        return {
            'symbol': symbol,
            'avg_return': avg_return,
            'avg_sharpe': avg_sharpe,
            'avg_drawdown': avg_dd,
            'profitable_years': profitable,
            'total_years': len(valid_results),
            'cumulative': cumulative - 1,
            'annualized': cumulative**(1/len(valid_results)) - 1,
            'yearly_results': results
        }

    return None


def main():
    print("=" * 60)
    print("七姐妹 平均阈值策略测试")
    print("=" * 60)
    print("\n方法: 4年各自找最优阈值 → 取平均 → 测试第5年")

    loader = DataLoader(db_config)

    all_results = []
    for symbol in MAG7_SYMBOLS:
        result = test_symbol_avg_threshold(symbol, loader)
        if result:
            all_results.append(result)

    loader.close()

    # 总汇总
    print("\n" + "=" * 60)
    print("七姐妹汇总对比 (平均阈值法)")
    print("=" * 60)

    print(f"\n{'股票':<8} {'平均收益':>10} {'夏普':>8} {'回撤':>10} {'盈利年':>8} {'5年累计':>12} {'年化':>10}")
    print("-" * 75)

    for r in sorted(all_results, key=lambda x: x['cumulative'], reverse=True):
        print(f"{r['symbol']:<8} {r['avg_return']:>10.2%} {r['avg_sharpe']:>8.2f} "
              f"{r['avg_drawdown']:>10.2%} {r['profitable_years']:>6}/{r['total_years']} "
              f"{r['cumulative']:>12.2%} {r['annualized']:>10.2%}")

    print("-" * 75)
    if all_results:
        total_avg = np.mean([r['avg_return'] for r in all_results])
        total_sharpe = np.mean([r['avg_sharpe'] for r in all_results])
        total_cum = np.mean([r['cumulative'] for r in all_results])
        print(f"{'平均':<8} {total_avg:>10.2%} {total_sharpe:>8.2f} {'':<10} {'':<8} {total_cum:>12.2%}")

    # 年度收益对比表
    print("\n" + "=" * 60)
    print("年度收益对比")
    print("=" * 60)

    years = [2021, 2022, 2023, 2024, 2025]
    header = f"{'股票':<8}"
    for year in years:
        header += f" {year:>10}"
    header += f" {'累计':>12}"
    print(header)
    print("-" * 75)

    for r in all_results:
        row = f"{r['symbol']:<8}"
        for yr in r['yearly_results']:
            row += f" {yr['return']:>10.2%}"
        row += f" {r['cumulative']:>12.2%}"
        print(row)

    # 与固定阈值对比
    print("\n" + "=" * 60)
    print("平均阈值法 vs 固定阈值 <-10, >30 对比")
    print("=" * 60)

    # 固定阈值结果 (从之前的测试)
    fixed_results = {
        'NVDA': {'cumulative': 5.1413, 'annualized': 0.4376},
        'TSLA': {'cumulative': 3.3637, 'annualized': 0.3427},
        'AAPL': {'cumulative': 1.2750, 'annualized': 0.1787},
        'GOOGL': {'cumulative': 1.2239, 'annualized': 0.1733},
        'MSFT': {'cumulative': 0.9640, 'annualized': 0.1445},
        'AMZN': {'cumulative': 0.4717, 'annualized': 0.0803},
        'META': {'cumulative': 0.4342, 'annualized': 0.0748},
    }

    print(f"\n{'股票':<8} {'固定阈值累计':>14} {'平均阈值累计':>14} {'差异':>10} {'更优':>8}")
    print("-" * 60)

    for r in all_results:
        symbol = r['symbol']
        if symbol in fixed_results:
            fixed_cum = fixed_results[symbol]['cumulative']
            avg_cum = r['cumulative']
            diff = avg_cum - fixed_cum
            better = "平均阈值" if diff > 0 else "固定阈值"
            print(f"{symbol:<8} {fixed_cum:>14.2%} {avg_cum:>14.2%} {diff:>+10.2%} {better:>8}")

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
