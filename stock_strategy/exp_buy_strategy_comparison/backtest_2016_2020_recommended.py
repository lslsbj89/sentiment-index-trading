"""
2016-2020年度回测 - 使用2026推荐参数
=====================================
测试推荐参数在历史数据上的适用性（样本外测试）
"""

import sys
sys.path.insert(0, '/Users/sc2025/Desktop/Claude/Quantrade/src')

import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime
from data_loader import DataLoader

# ============================================================
# 配置
# ============================================================

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "crypto_fear_greed_2",
    "user": "sc2025",
    "password": ""
}

INITIAL_CAPITAL = 100000
COMMISSION = 0.001
SLIPPAGE = 0.001

# 统一买入阈值
BUY_THRESHOLDS = [5, 0, -5, -10]
BATCH_PCT = 0.25

# 2026推荐的卖出参数 (来自推荐文档)
RECOMMENDED_PARAMS = {
    "NVDA": {"and_threshold": 25, "or_threshold": 60},
    "TSLA": {"and_threshold": 25, "or_threshold": 50},
    "GOOGL": {"and_threshold": 15, "or_threshold": 30},
    "AAPL": {"and_threshold": 10, "or_threshold": 30},
    "MSFT": {"and_threshold": 15, "or_threshold": 30},
    "META": {"and_threshold": 15, "or_threshold": 55},
    "AMZN": {"and_threshold": 20, "or_threshold": 30},
}

# 回测期 (样本外测试)
TEST_START = "2016-01-01"
TEST_END = "2020-12-31"


def load_sentiment_s3(symbol):
    conn = psycopg2.connect(**DB_CONFIG)
    query = """
        SELECT date, smoothed_index
        FROM fear_greed_index_s3
        WHERE symbol = %s
        ORDER BY date
    """
    df = pd.read_sql(query, conn, params=(symbol,))
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.set_index('date')
    conn.close()
    return df


def load_price(symbol):
    loader = DataLoader(DB_CONFIG)
    price_df = loader.load_ohlcv(symbol, start_date="2015-01-01")
    loader.close()
    return price_df


def prepare_data(price_df, sentiment_df, start_date, end_date):
    df = price_df.copy()
    df['sentiment'] = sentiment_df['smoothed_index'].reindex(df.index)
    df['MA50'] = df['Close'].rolling(50).mean()
    df = df.dropna()
    df = df[start_date:end_date]
    return df


def run_backtest(df, and_threshold, or_threshold):
    """运行单次回测"""
    cash = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    bought_levels = set()
    initial_capital_for_batch = INITIAL_CAPITAL

    trades = []
    portfolio_history = []
    yearly_values = {}

    for i in range(len(df)):
        current_date = df.index[i]
        current_price = df['Close'].iloc[i]
        current_sentiment = df['sentiment'].iloc[i]
        current_ma50 = df['MA50'].iloc[i]

        # 记录每日组合价值
        daily_value = cash + position * current_price
        portfolio_history.append({
            'date': current_date,
            'cash': cash,
            'position': position,
            'price': current_price,
            'value': daily_value,
            'sentiment': current_sentiment
        })

        # 记录年末资产
        year = current_date.year
        yearly_values[year] = daily_value

        # ========== 卖出逻辑 ==========
        if position > 0:
            sell_signal = False
            sell_reason = ""

            if current_sentiment > or_threshold:
                sell_signal = True
                sell_reason = f"OR: sent {current_sentiment:.1f} > {or_threshold}"
            elif current_sentiment > and_threshold and current_price < current_ma50:
                sell_signal = True
                sell_reason = f"AND: sent {current_sentiment:.1f} > {and_threshold} & P<MA50"

            if sell_signal:
                sell_price = current_price * (1 - SLIPPAGE) * (1 - COMMISSION)
                profit_pct = (sell_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
                sell_value = position * sell_price

                trades.append({
                    'date': current_date,
                    'type': 'SELL',
                    'price': current_price,
                    'shares': position,
                    'value': sell_value,
                    'sentiment': current_sentiment,
                    'reason': sell_reason,
                    'profit_pct': profit_pct
                })

                cash += sell_value
                position = 0
                entry_price = 0
                bought_levels = set()
                initial_capital_for_batch = cash

        # ========== 买入逻辑 ==========
        for level_idx, threshold in enumerate(BUY_THRESHOLDS):
            if level_idx not in bought_levels and current_sentiment < threshold:
                buy_price = current_price * (1 + SLIPPAGE) * (1 + COMMISSION)
                target_value = initial_capital_for_batch * BATCH_PCT
                shares = int(target_value / buy_price)

                if shares > 0 and cash >= shares * buy_price:
                    buy_cost = shares * buy_price

                    if position > 0:
                        total_cost = entry_price * position + buy_cost
                        position += shares
                        entry_price = total_cost / position
                    else:
                        position = shares
                        entry_price = buy_price

                    cash -= buy_cost
                    bought_levels.add(level_idx)

                    trades.append({
                        'date': current_date,
                        'type': 'BUY',
                        'price': current_price,
                        'shares': shares,
                        'value': buy_cost,
                        'sentiment': current_sentiment,
                        'reason': f"Batch{level_idx+1}: sent {current_sentiment:.1f} < {threshold}",
                        'batch': level_idx + 1
                    })

    # 计算最终结果
    final_value = cash + position * df['Close'].iloc[-1]
    total_return = (final_value / INITIAL_CAPITAL - 1) * 100

    # 计算最大回撤
    portfolio_df = pd.DataFrame(portfolio_history)
    portfolio_df['peak'] = portfolio_df['value'].expanding().max()
    portfolio_df['drawdown'] = (portfolio_df['value'] - portfolio_df['peak']) / portfolio_df['peak'] * 100
    max_drawdown = portfolio_df['drawdown'].min()

    # 计算买入持有收益
    buy_hold_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100

    # 计算年化收益
    years = (df.index[-1] - df.index[0]).days / 365.25
    annualized_return = ((final_value / INITIAL_CAPITAL) ** (1/years) - 1) * 100 if years > 0 else 0
    annualized_buyhold = ((1 + buy_hold_return/100) ** (1/years) - 1) * 100 if years > 0 else 0

    # 计算年度收益
    yearly_returns = {}
    years_list = sorted(yearly_values.keys())
    for i, year in enumerate(years_list):
        if i == 0:
            yearly_returns[year] = (yearly_values[year] / INITIAL_CAPITAL - 1) * 100
        else:
            prev_year = years_list[i-1]
            yearly_returns[year] = (yearly_values[year] / yearly_values[prev_year] - 1) * 100

    return {
        'final_value': final_value,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_buyhold': annualized_buyhold,
        'max_drawdown': max_drawdown,
        'buy_hold_return': buy_hold_return,
        'num_trades': len(trades),
        'num_buys': len([t for t in trades if t['type'] == 'BUY']),
        'num_sells': len([t for t in trades if t['type'] == 'SELL']),
        'trades': trades,
        'portfolio': portfolio_df,
        'yearly_returns': yearly_returns,
        'yearly_values': yearly_values,
        'end_position': position,
        'end_cash': cash
    }


def main():
    print("=" * 100)
    print("  2016-2020年度回测 - 使用2026推荐参数 (样本外测试)")
    print("=" * 100)
    print(f"\n  回测期间: {TEST_START} ~ {TEST_END} (5年)")
    print(f"  测试目的: 验证推荐参数在历史数据上的适用性")
    print(f"  初始资金: ${INITIAL_CAPITAL:,}")
    print(f"  买入策略: 阈值分批 (sent < 5/0/-5/-10, 每档25%)")
    print(f"  卖出策略: OR条件 或 AND条件(需price<MA50)")
    print()

    results = []

    for symbol in RECOMMENDED_PARAMS.keys():
        params = RECOMMENDED_PARAMS[symbol]
        and_t = params['and_threshold']
        or_t = params['or_threshold']

        print(f"\n{'─'*100}")
        print(f"  {symbol} (参数: AND>{and_t}, OR>{or_t})")
        print(f"{'─'*100}")

        # 加载数据
        price_df = load_price(symbol)
        sentiment_df = load_sentiment_s3(symbol)
        test_df = prepare_data(price_df, sentiment_df, TEST_START, TEST_END)

        if len(test_df) < 10:
            print(f"  ⚠️ 数据不足，跳过")
            continue

        # 运行回测
        result = run_backtest(test_df, and_t, or_t)
        result['symbol'] = symbol
        result['and_threshold'] = and_t
        result['or_threshold'] = or_t
        results.append(result)

        # 打印年度收益
        print(f"\n  年度收益:")
        print(f"    {'年份':<8} {'年末资产':>15} {'年度收益':>12}")
        print(f"    {'-'*40}")
        for year in sorted(result['yearly_returns'].keys()):
            print(f"    {year:<8} ${result['yearly_values'][year]:>13,.2f} {result['yearly_returns'][year]:>+11.2f}%")

        # 打印交易统计
        sells = [t for t in result['trades'] if t['type'] == 'SELL']
        print(f"\n  交易统计:")
        print(f"    总交易: {result['num_trades']}次 (买{result['num_buys']}次, 卖{result['num_sells']}次)")
        if sells:
            avg_profit = np.mean([t['profit_pct'] for t in sells])
            win_trades = [t for t in sells if t['profit_pct'] > 0]
            print(f"    卖出胜率: {len(win_trades)}/{len(sells)} ({len(win_trades)/len(sells)*100:.0f}%)")
            print(f"    平均卖出收益: {avg_profit:+.1f}%")

        # 打印结果
        print(f"\n  结果:")
        print(f"    期初: ${INITIAL_CAPITAL:,}")
        print(f"    期末: ${result['final_value']:,.2f}")
        print(f"    策略总收益: {result['total_return']:+.2f}%")
        print(f"    买入持有: {result['buy_hold_return']:+.2f}%")
        print(f"    超额收益: {result['total_return'] - result['buy_hold_return']:+.2f}%")
        print(f"    年化收益: {result['annualized_return']:+.2f}% (买入持有: {result['annualized_buyhold']:+.2f}%)")
        print(f"    最大回撤: {result['max_drawdown']:.2f}%")

    # 汇总表
    print(f"\n\n{'═'*100}")
    print(f"  2016-2020 五年回测汇总 (样本外测试)")
    print(f"{'═'*100}")

    print(f"\n  {'股票':<8} {'参数':<14} {'总收益':>12} {'年化':>10} {'买入持有':>12} {'超额':>12} {'最大回撤':>12} {'交易':>6}")
    print(f"  {'-'*95}")

    total_strategy = 0
    total_buyhold = 0

    for r in results:
        params_str = f"AND>{r['and_threshold']},OR>{r['or_threshold']}"
        excess = r['total_return'] - r['buy_hold_return']
        excess_marker = "✓" if excess > 0 else "✗"
        print(f"  {r['symbol']:<8} {params_str:<14} {r['total_return']:>+11.1f}% {r['annualized_return']:>+9.1f}% {r['buy_hold_return']:>+11.1f}% {excess:>+11.1f}% {excess_marker} {r['max_drawdown']:>11.1f}% {r['num_trades']:>6}")
        total_strategy += r['total_return']
        total_buyhold += r['buy_hold_return']

    print(f"  {'-'*95}")
    avg_strategy = total_strategy / len(results)
    avg_buyhold = total_buyhold / len(results)
    avg_excess = avg_strategy - avg_buyhold
    print(f"  {'平均':<8} {'':<14} {avg_strategy:>+11.1f}% {'':<10} {avg_buyhold:>+11.1f}% {avg_excess:>+11.1f}%")

    # 胜率统计
    win_count = sum(1 for r in results if r['total_return'] > r['buy_hold_return'])
    print(f"\n  超额收益胜率: {win_count}/{len(results)} ({win_count/len(results)*100:.0f}%)")

    # 年度表格
    print(f"\n\n{'═'*100}")
    print(f"  各股票年度收益对比")
    print(f"{'═'*100}")

    years = [2016, 2017, 2018, 2019, 2020]
    print(f"\n  {'股票':<8}", end="")
    for y in years:
        print(f" {y:>10}", end="")
    print(f" {'5年总计':>12}")
    print(f"  {'-'*75}")

    for r in results:
        print(f"  {r['symbol']:<8}", end="")
        for y in years:
            if y in r['yearly_returns']:
                print(f" {r['yearly_returns'][y]:>+9.1f}%", end="")
            else:
                print(f" {'N/A':>10}", end="")
        print(f" {r['total_return']:>+11.1f}%")

    # 2018和2020特殊年份分析
    print(f"\n\n{'═'*100}")
    print(f"  特殊年份表现分析")
    print(f"{'═'*100}")

    print(f"\n  2018年 (市场调整):")
    print(f"  {'股票':<8} {'2018收益':>12} {'评估':>10}")
    print(f"  {'-'*35}")
    for r in results:
        ret_2018 = r['yearly_returns'].get(2018, 0)
        if ret_2018 > 0:
            status = "✓ 盈利"
        elif ret_2018 > -10:
            status = "⚠ 小亏"
        else:
            status = "✗ 较差"
        print(f"  {r['symbol']:<8} {ret_2018:>+11.1f}% {status:>10}")

    print(f"\n  2020年 (疫情+反弹):")
    print(f"  {'股票':<8} {'2020收益':>12} {'评估':>10}")
    print(f"  {'-'*35}")
    for r in results:
        ret_2020 = r['yearly_returns'].get(2020, 0)
        if ret_2020 > 50:
            status = "✓ 优秀"
        elif ret_2020 > 20:
            status = "✓ 良好"
        else:
            status = "⚠ 一般"
        print(f"  {r['symbol']:<8} {ret_2020:>+11.1f}% {status:>10}")

    # 组合模拟
    print(f"\n\n{'═'*100}")
    print(f"  组合模拟 (按推荐配置)")
    print(f"{'═'*100}")

    allocations = {
        "NVDA": 0.25,
        "TSLA": 0.15,
        "GOOGL": 0.15,
        "AAPL": 0.15,
        "MSFT": 0.10,
        "META": 0.10,
    }

    portfolio_return = 0
    buyhold_return = 0

    print(f"\n  {'股票':<8} {'配置':>8} {'策略收益':>14} {'贡献':>12}")
    print(f"  {'-'*50}")

    for symbol, weight in allocations.items():
        r = next((x for x in results if x['symbol'] == symbol), None)
        if r:
            contrib = weight * r['total_return']
            bh_contrib = weight * r['buy_hold_return']
            portfolio_return += contrib
            buyhold_return += bh_contrib
            print(f"  {symbol:<8} {weight*100:>7.0f}% {r['total_return']:>+13.1f}% {contrib:>+11.1f}%")

    print(f"  {'-'*50}")
    print(f"  {'组合总计':<8} {sum(allocations.values())*100:>7.0f}% {portfolio_return:>+13.1f}%")
    print(f"  {'买入持有':<8} {'':<8} {buyhold_return:>+13.1f}%")
    print(f"  {'超额收益':<8} {'':<8} {portfolio_return - buyhold_return:>+13.1f}%")

    # 年化计算
    portfolio_annualized = ((1 + portfolio_return/100) ** (1/5) - 1) * 100
    buyhold_annualized = ((1 + buyhold_return/100) ** (1/5) - 1) * 100

    print(f"\n  年化收益:")
    print(f"    策略组合: {portfolio_annualized:+.2f}%/年")
    print(f"    买入持有: {buyhold_annualized:+.2f}%/年")

    # 换算成金额
    print(f"\n  以$100,000本金计算 (5年):")
    print(f"    策略组合: ${100000 * (1 + portfolio_return/100):,.2f}")
    print(f"    买入持有: ${100000 * (1 + buyhold_return/100):,.2f}")
    print(f"    超额收益: ${100000 * (portfolio_return - buyhold_return)/100:,.2f}")

    # 结论
    print(f"\n\n{'═'*100}")
    print(f"  样本外测试结论")
    print(f"{'═'*100}")

    print(f"""
  测试目的: 验证2026推荐参数在2016-2020历史数据上的表现

  结果分析:
  - 超额收益胜率: {win_count}/{len(results)} ({win_count/len(results)*100:.0f}%)
  - 平均超额收益: {avg_excess:+.1f}%
  - 组合超额收益: {portfolio_return - buyhold_return:+.1f}%

  参数适用性评估:
""")

    if win_count >= 5:
        print("  ✓ 推荐参数在样本外数据上表现良好，具有较强的泛化能力")
    elif win_count >= 4:
        print("  ⚠ 推荐参数在样本外数据上表现一般，部分股票适用")
    else:
        print("  ✗ 推荐参数在样本外数据上表现较差，可能存在过拟合")


if __name__ == "__main__":
    main()
