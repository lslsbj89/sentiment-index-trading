"""
七姐妹 (Magnificent 7) 情绪策略测试
固定阈值: buy < -10, sell > 30
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import psycopg2

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

# 固定阈值
BUY_THRESHOLD = -10
SELL_THRESHOLD = 30

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

# 测试周期
TEST_YEARS = [2021, 2022, 2023, 2024, 2025]


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


def run_backtest(stock_data, sentiment_data, start_date, end_date):
    mask = (stock_data.index >= start_date) & (stock_data.index <= end_date)
    price_data = stock_data[mask]

    if sentiment_data is None:
        return None, None, None

    sent_data = sentiment_data.reindex(price_data.index)

    if len(price_data) == 0:
        return None, None, None

    signals = pd.DataFrame(index=price_data.index)
    signals['smoothed_index'] = sent_data['smoothed_index']
    signals['buy_signal'] = (signals['smoothed_index'] < BUY_THRESHOLD).astype(int)
    signals['sell_signal'] = (signals['smoothed_index'] > SELL_THRESHOLD).astype(int)
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


def test_symbol(symbol, loader):
    """测试单个股票"""
    print(f"\n{'='*70}")
    print(f"股票: {symbol}")
    print(f"{'='*70}")

    # 加载数据
    try:
        stock_data = loader.load_ohlcv(symbol, start_date="2014-01-01")
        if len(stock_data) == 0:
            print(f"  ❌ 无价格数据")
            return None
    except Exception as e:
        print(f"  ❌ 加载价格数据失败: {e}")
        return None

    sentiment_data = load_sentiment_data(db_config, symbol)
    if sentiment_data is None or len(sentiment_data) == 0:
        print(f"  ❌ 无情绪数据")
        return None

    print(f"  价格数据: {len(stock_data)} 条 ({stock_data.index[0].strftime('%Y-%m-%d')} ~ {stock_data.index[-1].strftime('%Y-%m-%d')})")
    print(f"  情绪数据: {len(sentiment_data)} 条")

    # 测试每年
    results = []
    for year in TEST_YEARS:
        test_start = f"{year}-01-01"
        test_end = f"{year}-12-31"

        portfolio, metrics, trades = run_backtest(
            stock_data, sentiment_data, test_start, test_end
        )

        if metrics:
            ret = metrics.get('total_return', 0) or 0
            sharpe = metrics.get('sharpe_ratio', 0) or 0
            dd = metrics.get('max_drawdown', 0) or 0
            n_trades = metrics.get('total_trades', 0)

            results.append({
                'year': year,
                'return': ret,
                'sharpe': sharpe,
                'drawdown': dd,
                'trades': n_trades
            })
        else:
            results.append({
                'year': year,
                'return': 0,
                'sharpe': 0,
                'drawdown': 0,
                'trades': 0
            })

    # 打印年度结果
    print(f"\n  {'年份':<6} {'收益率':>10} {'夏普':>8} {'回撤':>10} {'交易':>6}")
    print(f"  {'-'*45}")
    for r in results:
        print(f"  {r['year']:<6} {r['return']:>10.2%} {r['sharpe']:>8.2f} {r['drawdown']:>10.2%} {r['trades']:>6}")

    # 计算汇总
    avg_return = np.mean([r['return'] for r in results])
    avg_sharpe = np.mean([r['sharpe'] for r in results])
    avg_dd = np.mean([r['drawdown'] for r in results])
    profitable_years = sum(1 for r in results if r['return'] > 0)

    # 累计收益
    cumulative = 1.0
    for r in results:
        cumulative *= (1 + r['return'])

    print(f"  {'-'*45}")
    print(f"  {'平均':<6} {avg_return:>10.2%} {avg_sharpe:>8.2f} {avg_dd:>10.2%}")
    print(f"\n  盈利年份: {profitable_years}/5")
    print(f"  5年累计: {(cumulative-1):.2%}")
    print(f"  年化收益: {(cumulative**(1/5)-1):.2%}")

    return {
        'symbol': symbol,
        'avg_return': avg_return,
        'avg_sharpe': avg_sharpe,
        'avg_drawdown': avg_dd,
        'profitable_years': profitable_years,
        'cumulative': cumulative - 1,
        'annualized': cumulative**(1/5) - 1,
        'yearly_results': results
    }


def main():
    print("=" * 70)
    print("七姐妹 (Magnificent 7) 情绪策略测试")
    print("=" * 70)
    print(f"\n固定阈值: buy < {BUY_THRESHOLD}, sell > {SELL_THRESHOLD}")
    print(f"测试周期: 2021-2025")

    loader = DataLoader(db_config)

    all_results = []
    for symbol in MAG7_SYMBOLS:
        result = test_symbol(symbol, loader)
        if result:
            all_results.append(result)

    loader.close()

    # 总汇总
    print("\n" + "=" * 70)
    print("七姐妹汇总对比")
    print("=" * 70)

    print(f"\n{'股票':<8} {'平均收益':>10} {'平均夏普':>10} {'平均回撤':>10} {'盈利年':>8} {'5年累计':>10} {'年化':>10}")
    print("-" * 75)

    for r in sorted(all_results, key=lambda x: x['cumulative'], reverse=True):
        print(f"{r['symbol']:<8} {r['avg_return']:>10.2%} {r['avg_sharpe']:>10.2f} "
              f"{r['avg_drawdown']:>10.2%} {r['profitable_years']:>6}/5 "
              f"{r['cumulative']:>10.2%} {r['annualized']:>10.2%}")

    print("-" * 75)

    # 总平均
    if all_results:
        total_avg_return = np.mean([r['avg_return'] for r in all_results])
        total_avg_sharpe = np.mean([r['avg_sharpe'] for r in all_results])
        total_profitable = sum(r['profitable_years'] for r in all_results)
        total_cumulative = np.mean([r['cumulative'] for r in all_results])

        print(f"{'平均':<8} {total_avg_return:>10.2%} {total_avg_sharpe:>10.2f} "
              f"{'':<10} {total_profitable:>6}/35 {total_cumulative:>10.2%}")

    # 年度对比表
    print("\n" + "=" * 70)
    print("年度收益对比")
    print("=" * 70)

    header = f"{'股票':<8}"
    for year in TEST_YEARS:
        header += f" {year:>10}"
    print(header)
    print("-" * 65)

    for r in all_results:
        row = f"{r['symbol']:<8}"
        for yr in r['yearly_results']:
            row += f" {yr['return']:>10.2%}"
        print(row)

    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
