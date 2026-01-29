"""
显示最优参数的交易详情
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import pandas as pd
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

# 最优参数
BUY_THRESHOLD = -15
AND_SELL_THRESHOLD = 25
OR_THRESHOLD = 45

TEST_START = "2021-01-01"
TEST_END = "2025-12-31"


def load_fear_greed_index(db_config, symbol):
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


def load_price_with_ma(db_config, symbol):
    loader = DataLoader(db_config)
    ohlcv = loader.load_ohlcv(symbol, start_date="2015-01-01")
    loader.close()
    ohlcv['MA50'] = ohlcv['Close'].rolling(window=50).mean()
    return ohlcv


def main():
    print("=" * 80)
    print(f"TSLA 交易详情 - 最优参数")
    print("=" * 80)
    print(f"\n策略规则:")
    print(f"  买入: smoothed_index < {BUY_THRESHOLD}")
    print(f"  卖出: (smoothed_index > {AND_SELL_THRESHOLD} AND 价格 < MA50) OR (smoothed_index > {OR_THRESHOLD})")

    # 加载数据
    sentiment_data = load_fear_greed_index(db_config, SYMBOL)
    price_data = load_price_with_ma(db_config, SYMBOL)

    # 筛选时间范围
    start_ts = pd.Timestamp(TEST_START, tz='UTC')
    end_ts = pd.Timestamp(TEST_END, tz='UTC')
    mask = (price_data.index >= start_ts) & (price_data.index <= end_ts)
    test_price = price_data[mask].copy()
    test_sentiment = sentiment_data.reindex(test_price.index)

    # 构建信号
    signals = pd.DataFrame(index=test_price.index)
    signals['smoothed_index'] = test_sentiment['smoothed_index']
    signals['Close'] = test_price['Close']
    signals['MA50'] = test_price['MA50']

    signals['buy_signal'] = (signals['smoothed_index'] < BUY_THRESHOLD).astype(int)
    and_condition = (signals['smoothed_index'] > AND_SELL_THRESHOLD) & (signals['Close'] < signals['MA50'])
    or_condition = signals['smoothed_index'] > OR_THRESHOLD
    signals['sell_signal'] = (and_condition | or_condition).astype(int)
    signals['prob_profit'] = 0.5
    signals['position_size'] = 0

    # 回测
    backtester = EnhancedBacktester(
        initial_capital=100000,
        commission_rate=0.001,
        slippage_rate=0.001,
        take_profit_pct=999.0,
        stop_loss_pct=999.0,
        max_holding_days=999,
        use_dynamic_position=True,
        position_pct=0.8
    )

    portfolio, metrics, trades = backtester.run_backtest_with_sell_signal(signals, test_price)

    # 显示交易详情
    print(f"\n{'=' * 80}")
    print(f"交易记录 (共 {len(trades)} 笔)")
    print("=" * 80)

    for i, trade in enumerate(trades, 1):
        entry_date = trade['entry_date']
        exit_date = trade['exit_date']
        entry_price = trade['entry_price']
        exit_price = trade['exit_price']
        shares = trade['shares']
        pnl = trade['profit']
        pnl_pct = trade['profit_pct'] * 100
        holding_days = trade['holding_days']
        exit_reason = trade['exit_reason']

        # 获取入场和出场时的情绪指数
        entry_sentiment = signals.loc[entry_date, 'smoothed_index'] if entry_date in signals.index else 'N/A'
        exit_sentiment = signals.loc[exit_date, 'smoothed_index'] if exit_date in signals.index else 'N/A'

        # 获取出场时的MA50
        exit_ma50 = signals.loc[exit_date, 'MA50'] if exit_date in signals.index else 'N/A'

        print(f"\n{'─' * 80}")
        print(f"交易 #{i}")
        print(f"{'─' * 80}")
        print(f"  入场日期: {entry_date.strftime('%Y-%m-%d')}")
        print(f"  入场价格: ${entry_price:.2f}")
        print(f"  入场情绪: {entry_sentiment:.2f}" if isinstance(entry_sentiment, float) else f"  入场情绪: {entry_sentiment}")
        print(f"  股数:     {shares}")
        print()
        print(f"  出场日期: {exit_date.strftime('%Y-%m-%d')}")
        print(f"  出场价格: ${exit_price:.2f}")
        print(f"  出场情绪: {exit_sentiment:.2f}" if isinstance(exit_sentiment, float) else f"  出场情绪: {exit_sentiment}")
        print(f"  出场MA50: ${exit_ma50:.2f}" if isinstance(exit_ma50, float) else f"  出场MA50: {exit_ma50}")
        print(f"  出场原因: {exit_reason}")
        print()
        print(f"  持仓天数: {holding_days} 天")
        print(f"  盈亏金额: ${pnl:+,.2f}")
        print(f"  盈亏比例: {pnl_pct:+.2f}%")

    # 汇总
    print(f"\n{'=' * 80}")
    print("汇总统计")
    print("=" * 80)
    print(f"  初始资金:   $100,000.00")
    print(f"  最终资金:   ${portfolio['total_value'].iloc[-1]:,.2f}")
    print(f"  总收益:     {metrics['total_return']*100:+.2f}%")
    print(f"  年化收益:   {metrics['annualized_return']*100:.2f}%")
    print(f"  夏普比率:   {metrics['sharpe_ratio']:.2f}")
    print(f"  最大回撤:   {metrics['max_drawdown']*100:.2f}%")
    print(f"  总交易数:   {len(trades)}")

    winning = sum(1 for t in trades if t['profit'] > 0)
    losing = sum(1 for t in trades if t['profit'] <= 0)
    print(f"  盈利交易:   {winning}")
    print(f"  亏损交易:   {losing}")


if __name__ == "__main__":
    main()
