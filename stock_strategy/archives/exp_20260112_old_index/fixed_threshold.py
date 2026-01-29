"""
固定阈值测试：buy<0, sell>30
无需训练、无需验证，直接测试
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import psycopg2

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
TEST_PERIODS = [
    {"year": 2021, "label": "2021 (复苏牛市)"},
    {"year": 2022, "label": "2022 (加息熊市)"},
    {"year": 2023, "label": "2023 (震荡反弹)"},
    {"year": 2024, "label": "2024 (AI牛市)"},
    {"year": 2025, "label": "2025 (科技牛市)"},
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


def run_backtest(stock_data, sentiment_data, start_date, end_date):
    mask = (stock_data.index >= start_date) & (stock_data.index <= end_date)
    price_data = stock_data[mask]
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


def main():
    print("=" * 70)
    print("TSLA 纯情绪策略 - 固定阈值测试")
    print("=" * 70)
    print(f"\n固定阈值: buy < {BUY_THRESHOLD}, sell > {SELL_THRESHOLD}")
    print("无止盈止损，纯情绪信号驱动")

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
        year = period["year"]
        test_start = f"{year}-01-01"
        test_end = f"{year}-12-31"

        print(f"\n{'='*70}")
        print(f"周期: {period['label']}")
        print(f"{'='*70}")

        portfolio, metrics, trades = run_backtest(
            stock_data, sentiment_data, test_start, test_end
        )

        if not metrics:
            print("  ❌ 无结果")
            continue

        test_return = metrics.get('total_return', 0) or 0
        test_sharpe = metrics.get('sharpe_ratio', 0) or 0
        test_dd = metrics.get('max_drawdown', 0) or 0
        n_trades = metrics.get('total_trades', 0)
        win_rate = metrics.get('trade_win_rate', 0) or 0

        print(f"  收益率: {test_return:.2%}")
        print(f"  夏普比率: {test_sharpe:.2f}")
        print(f"  最大回撤: {test_dd:.2%}")
        print(f"  交易次数: {n_trades}")
        print(f"  胜率: {win_rate:.2%}")

        if trades:
            print(f"\n  [交易明细]")
            for t in trades:
                entry = str(t['entry_date'])[:10]
                exit_d = str(t.get('exit_date', ''))[:10] if pd.notna(t.get('exit_date')) else 'Open'
                entry_p = t['entry_price']
                exit_p = t.get('exit_price', 0)
                ret_pct = (exit_p - entry_p) / entry_p if exit_p else 0
                reason = t.get('exit_reason', '-')
                print(f"    {entry} @${entry_p:.0f} → {exit_d} @${exit_p:.0f} | {ret_pct:+.1%} | {reason}")

        results.append({
            'year': year,
            'label': period['label'],
            'return': test_return,
            'sharpe': test_sharpe,
            'drawdown': test_dd,
            'trades': n_trades,
            'win_rate': win_rate
        })

    # 汇总
    print("\n" + "=" * 70)
    print("汇总结果")
    print("=" * 70)

    print(f"\n固定阈值: buy < {BUY_THRESHOLD}, sell > {SELL_THRESHOLD}")
    print(f"\n{'周期':<20} {'收益率':>10} {'夏普':>8} {'回撤':>10} {'交易':>6} {'胜率':>8}")
    print("-" * 70)

    for r in results:
        print(f"{r['label']:<20} {r['return']:>10.2%} {r['sharpe']:>8.2f} "
              f"{r['drawdown']:>10.2%} {r['trades']:>6} {r['win_rate']:>8.2%}")

    if results:
        avg_return = np.mean([r['return'] for r in results])
        avg_sharpe = np.mean([r['sharpe'] for r in results])
        avg_dd = np.mean([r['drawdown'] for r in results])
        total_trades = sum([r['trades'] for r in results])

        print("-" * 70)
        print(f"{'平均':<20} {avg_return:>10.2%} {avg_sharpe:>8.2f} {avg_dd:>10.2%} {total_trades:>6}")

        print("\n稳定性分析:")
        profitable = sum(1 for r in results if r['return'] > 0)
        print(f"  盈利年份: {profitable}/{len(results)}")
        print(f"  收益率标准差: {np.std([r['return'] for r in results]):.2%}")

        # 累计收益
        cumulative = 1.0
        for r in results:
            cumulative *= (1 + r['return'])
        print(f"  5年累计收益: {(cumulative - 1):.2%}")
        print(f"  年化收益: {(cumulative ** (1/5) - 1):.2%}")

    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
