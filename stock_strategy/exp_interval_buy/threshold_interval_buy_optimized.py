"""
间隔分批买入策略 - 联合优化版
=====================================
数据源: fear_greed_index_s3 (Smoothing=3, MoneyFlow=13)

实验目的:
  训练期和测试期统一使用间隔分批买入逻辑,
  联合优化买入参数 (buy_threshold, interval_days) 和卖出参数 (AND, OR)

买入逻辑 (训练+测试统一):
- 当 sentiment < buy_threshold → 触发买入模式, 首次买入20%
- 每隔 interval_days 天 → 再买入20%
- 当 sentiment 重新 >= buy_threshold → 一次性买入剩余全部仓位

卖出逻辑 (不变):
- OR条件: sentiment > OR阈值 → 无条件卖出
- AND条件: sentiment > AND阈值 且 price < MA50 → 卖出

网格搜索 (联合优化):
- buy_threshold: [3, 5, 7, 10]
- interval_days: [5, 7, 10, 14]
- and_threshold: [5, 10, 15, 20, 25]
- or_threshold: [30, 40, 50, 55, 60]
- 总组合: 4 × 4 × 5 × 5 = 400

Walk-Forward验证:
- 4年训练 + 1年测试
- 滚动窗口: W2020-W2025
- 连续资金传递
"""

import sys
sys.path.insert(0, '/Users/sc2025/Desktop/Claude/Quantrade/src')

import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime
from data_loader import DataLoader

# ============================================================
# 配置参数
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

BATCH_PCT = 0.20  # 每次买入20%, 固定不搜索

# 联合搜索范围
BUY_THRESHOLD_RANGE = [3, 5, 7, 10]
INTERVAL_DAYS_RANGE = [5, 7, 10, 14]
AND_SELL_RANGE = [5, 10, 15, 20, 25]
OR_SELL_RANGE = [30, 40, 50, 55, 60]

WINDOWS = [
    {"name": "W2020", "train": ("2016-01-01", "2019-12-31"), "test": ("2020-01-01", "2020-12-31")},
    {"name": "W2021", "train": ("2017-01-01", "2020-12-31"), "test": ("2021-01-01", "2021-12-31")},
    {"name": "W2022", "train": ("2018-01-01", "2021-12-31"), "test": ("2022-01-01", "2022-12-31")},
    {"name": "W2023", "train": ("2019-01-01", "2022-12-31"), "test": ("2023-01-01", "2023-12-31")},
    {"name": "W2024", "train": ("2020-01-01", "2023-12-31"), "test": ("2024-01-01", "2024-12-31")},
    {"name": "W2025", "train": ("2021-01-01", "2024-12-31"), "test": ("2025-01-01", "2025-12-31")},
]

SYMBOLS = ["NVDA"]


# ============================================================
# 数据加载
# ============================================================

def load_sentiment_s3(symbol):
    """从 fear_greed_index_s3 表加载情绪数据"""
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
    """加载价格数据"""
    loader = DataLoader(DB_CONFIG)
    price_df = loader.load_ohlcv(symbol, start_date="2011-01-01")
    loader.close()
    return price_df


def prepare_data(price_df, sentiment_df, start_date, end_date):
    """准备回测数据"""
    df = price_df.copy()
    df['sentiment'] = sentiment_df['smoothed_index'].reindex(df.index)
    df['MA50'] = df['Close'].rolling(50).mean()
    df = df.dropna()
    df = df[start_date:end_date]
    return df


# ============================================================
# 间隔分批买入策略 (训练+测试统一)
# ============================================================

def run_interval_buy(df, buy_threshold, interval_days, and_threshold, or_threshold,
                     cash, position, entry_price,
                     in_buy_mode=False, last_buy_date=None, buy_base_capital=None,
                     batches_bought=0, record_trades=True):
    """
    间隔分批买入策略

    Parameters:
    -----------
    buy_threshold : float   买入触发阈值
    interval_days : int     间隔天数
    and_threshold : float   AND卖出阈值
    or_threshold : float    OR卖出阈值
    record_trades : bool    是否记录交易详情 (网格搜索时关闭以提速)
    """
    trades = [] if record_trades else None
    portfolio_values = [] if record_trades else None

    for i in range(len(df)):
        current_date = df.index[i]
        current_price = df['Close'].iloc[i]
        current_sentiment = df['sentiment'].iloc[i]
        current_ma50 = df['MA50'].iloc[i]

        # ========== 卖出逻辑 ==========
        if position > 0:
            sell_signal = False
            sell_reason = ""

            if current_sentiment > or_threshold:
                sell_signal = True
                sell_reason = f"OR: sentiment {current_sentiment:.1f} > {or_threshold}"
            elif current_sentiment > and_threshold and current_price < current_ma50:
                sell_signal = True
                sell_reason = f"AND: sentiment {current_sentiment:.1f} > {and_threshold} & price < MA50"

            if sell_signal:
                sell_price = current_price * (1 - SLIPPAGE) * (1 - COMMISSION)
                profit_pct = (sell_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
                sell_value = position * sell_price
                cash += sell_value

                if record_trades:
                    trades.append({
                        'type': 'SELL', 'date': current_date, 'price': current_price,
                        'shares': position, 'value': sell_value, 'sentiment': current_sentiment,
                        'reason': sell_reason, 'profit_pct': profit_pct
                    })

                position = 0
                entry_price = 0
                in_buy_mode = False
                last_buy_date = None
                buy_base_capital = None
                batches_bought = 0

        # ========== 买入逻辑: 间隔分批 ==========
        if position == 0 or in_buy_mode:
            if not in_buy_mode and current_sentiment < buy_threshold:
                in_buy_mode = True
                buy_base_capital = cash + position * current_price
                batches_bought = 0
                last_buy_date = None

            if in_buy_mode:
                should_buy = False
                buy_reason = ""
                buy_all_remaining = False

                if last_buy_date is None:
                    should_buy = True
                    buy_reason = f"Interval B1: sentiment {current_sentiment:.1f} < {buy_threshold} (enter)"
                elif current_sentiment >= buy_threshold:
                    should_buy = True
                    buy_all_remaining = True
                    buy_reason = f"Recovery ALL: sentiment {current_sentiment:.1f} >= {buy_threshold}"
                elif (current_date - last_buy_date).days >= interval_days:
                    should_buy = True
                    batches_bought += 1
                    buy_reason = f"Interval B{batches_bought+1}: {(current_date - last_buy_date).days}d elapsed"

                if should_buy:
                    buy_price = current_price * (1 + SLIPPAGE) * (1 + COMMISSION)

                    if buy_all_remaining:
                        shares = int(cash / buy_price)
                    else:
                        target_value = buy_base_capital * BATCH_PCT
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
                        last_buy_date = current_date

                        if record_trades:
                            trades.append({
                                'type': 'BUY', 'date': current_date, 'price': current_price,
                                'shares': shares, 'value': buy_cost, 'sentiment': current_sentiment,
                                'reason': buy_reason, 'batch': batches_bought + 1
                            })

                    if buy_all_remaining:
                        in_buy_mode = False
                        buy_base_capital = None
                        batches_bought = 0

        if record_trades:
            total_value = cash + position * current_price
            portfolio_values.append({
                'date': current_date, 'value': total_value,
                'cash': cash, 'position': position, 'price': current_price
            })

    final_value = cash + position * df['Close'].iloc[-1] if len(df) > 0 else cash

    return (final_value, trades, cash, position, entry_price,
            in_buy_mode, last_buy_date, buy_base_capital, batches_bought)


# ============================================================
# 联合网格搜索
# ============================================================

def grid_search(train_df):
    """联合优化: buy_threshold × interval_days × AND × OR"""
    best_return = -float('inf')
    best_params = None
    total = len(BUY_THRESHOLD_RANGE) * len(INTERVAL_DAYS_RANGE) * len(AND_SELL_RANGE) * len(OR_SELL_RANGE)

    for buy_t in BUY_THRESHOLD_RANGE:
        for intv in INTERVAL_DAYS_RANGE:
            for and_t in AND_SELL_RANGE:
                for or_t in OR_SELL_RANGE:
                    final_value, _, _, _, _, _, _, _, _ = run_interval_buy(
                        train_df, buy_t, intv, and_t, or_t,
                        INITIAL_CAPITAL, 0, 0,
                        record_trades=False
                    )
                    ret = (final_value / INITIAL_CAPITAL - 1) * 100

                    if ret > best_return:
                        best_return = ret
                        best_params = (buy_t, intv, and_t, or_t)

    return best_params, best_return


# ============================================================
# Walk-Forward 分析
# ============================================================

def run_walk_forward(symbol):
    """对单个股票运行Walk-Forward分析"""
    print(f"\n{'='*70}")
    print(f"  {symbol} - 间隔分批买入 联合优化 Walk-Forward 分析")
    print(f"{'='*70}")

    price_df = load_price(symbol)
    sentiment_df = load_sentiment_s3(symbol)

    print(f"  价格数据: {price_df.index.min().date()} ~ {price_df.index.max().date()}")
    print(f"  情绪数据: {sentiment_df.index.min().date()} ~ {sentiment_df.index.max().date()}")
    print(f"  搜索空间: buy_t{BUY_THRESHOLD_RANGE} × intv{INTERVAL_DAYS_RANGE} × AND{AND_SELL_RANGE} × OR{OR_SELL_RANGE}")
    print(f"  总组合: {len(BUY_THRESHOLD_RANGE)*len(INTERVAL_DAYS_RANGE)*len(AND_SELL_RANGE)*len(OR_SELL_RANGE)}")

    cash = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    in_buy_mode = False
    last_buy_date = None
    buy_base_capital = None
    batches_bought = 0

    results = []
    all_trades = []

    for window in WINDOWS:
        window_name = window['name']
        train_start, train_end = window['train']
        test_start, test_end = window['test']

        train_df = prepare_data(price_df, sentiment_df, train_start, train_end)
        test_df = prepare_data(price_df, sentiment_df, test_start, test_end)

        if len(train_df) < 100 or len(test_df) < 10:
            print(f"\n  {window_name}: 数据不足，跳过")
            continue

        # 训练期: 联合网格搜索
        import time
        t0 = time.time()
        best_params, train_return = grid_search(train_df)
        buy_t, intv, and_t, or_t = best_params
        search_time = time.time() - t0

        test_start_value = cash + position * test_df['Close'].iloc[0] if len(test_df) > 0 else cash

        # 测试期: 使用训练出的最优参数
        (final_value, trades, cash, position, entry_price,
         in_buy_mode, last_buy_date, buy_base_capital, batches_bought) = run_interval_buy(
            test_df, buy_t, intv, and_t, or_t,
            cash, position, entry_price,
            in_buy_mode, last_buy_date, buy_base_capital, batches_bought,
            record_trades=True
        )

        test_return = (final_value / test_start_value - 1) * 100 if test_start_value > 0 else 0

        buy_trades = [t for t in trades if t['type'] == 'BUY']
        sell_trades = [t for t in trades if t['type'] == 'SELL']

        results.append({
            'window': window_name,
            'buy_threshold': buy_t,
            'interval_days': intv,
            'and_threshold': and_t,
            'or_threshold': or_t,
            'train_return': train_return,
            'test_return': test_return,
            'num_buys': len(buy_trades),
            'num_sells': len(sell_trades),
            'end_value': final_value
        })

        for t in trades:
            t['window'] = window_name
            t['symbol'] = symbol
            all_trades.append(t)

        print(f"\n  {window_name}: Train {train_start[:4]}-{train_end[:4]} → Test {test_start[:4]} ({search_time:.1f}s)")
        print(f"    最优参数: buy<{buy_t}, intv={intv}d, AND>{and_t}, OR>{or_t} → 训练+{train_return:.1f}%")
        print(f"    测试: {test_return:+.1f}% | 买入{len(buy_trades)}次, 卖出{len(sell_trades)}次")

        if trades:
            for t in trades:
                if t['type'] == 'BUY':
                    print(f"      {t['date'].strftime('%Y-%m-%d')} BUY  ${t['price']:.2f} x{t['shares']} | {t['reason']}")
                else:
                    print(f"      {t['date'].strftime('%Y-%m-%d')} SELL ${t['price']:.2f} x{t['shares']} | {t['profit_pct']:+.1f}% | {t['reason']}")

    total_return = (final_value / INITIAL_CAPITAL - 1) * 100
    print(f"\n  总收益: ${INITIAL_CAPITAL:,} → ${final_value:,.0f} ({total_return:+.1f}%)")

    all_sells = [t for t in all_trades if t['type'] == 'SELL']
    if all_sells:
        win_rate = len([t for t in all_sells if t['profit_pct'] > 0]) / len(all_sells) * 100
    else:
        win_rate = 0
    print(f"  胜率: {win_rate:.1f}% ({len([t for t in all_sells if t['profit_pct'] > 0])}/{len(all_sells)})")

    return {
        'symbol': symbol,
        'total_return': total_return,
        'final_value': final_value,
        'win_rate': win_rate,
        'results': results,
        'trades': all_trades,
        'recommended_params': results[-1] if results else None
    }


# ============================================================
# 主函数
# ============================================================

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("="*70)
    print("间隔分批买入策略 - 联合优化 Walk-Forward 验证")
    print("="*70)
    print(f"搜索范围:")
    print(f"  buy_threshold: {BUY_THRESHOLD_RANGE}")
    print(f"  interval_days: {INTERVAL_DAYS_RANGE}")
    print(f"  batch_pct:     {BATCH_PCT*100:.0f}% (固定)")
    print(f"  and_threshold: {AND_SELL_RANGE}")
    print(f"  or_threshold:  {OR_SELL_RANGE}")
    total = len(BUY_THRESHOLD_RANGE)*len(INTERVAL_DAYS_RANGE)*len(AND_SELL_RANGE)*len(OR_SELL_RANGE)
    print(f"  总组合: {total}")
    print(f"初始资金: ${INITIAL_CAPITAL:,}")

    all_results = []
    all_trades = []
    recommendations = []

    for symbol in SYMBOLS:
        result = run_walk_forward(symbol)
        all_results.append(result)
        all_trades.extend(result['trades'])

        if result['recommended_params']:
            r = result['recommended_params']
            recommendations.append({
                'symbol': symbol,
                'buy_threshold': r['buy_threshold'],
                'interval_days': r['interval_days'],
                'and_threshold': r['and_threshold'],
                'or_threshold': r['or_threshold'],
                'total_return': result['total_return'],
                'win_rate': result['win_rate']
            })

    # 打印汇总
    print("\n" + "="*70)
    print("最终排名 (2020-2025 连续回测)")
    print("="*70)

    all_results.sort(key=lambda x: x['total_return'], reverse=True)
    print(f"\n{'排名':<4} {'股票':<8} {'最终资产':>14} {'总收益':>12} {'胜率':>10}")
    print("-"*55)
    for i, r in enumerate(all_results, 1):
        print(f"{i:<4} {r['symbol']:<8} ${r['final_value']:>12,.0f} {r['total_return']:>+10.1f}% {r['win_rate']:>9.1f}%")

    # 2026推荐参数
    if recommendations:
        print("\n" + "="*70)
        print("2026年推荐参数 (基于2021-2024训练)")
        print("="*70)

        rec_df = pd.DataFrame(recommendations)
        rec_df = rec_df.sort_values('total_return', ascending=False)

        print(f"\n{'股票':<8} {'buy<':>6} {'intv':>6} {'AND>':>6} {'OR>':>6} {'总收益':>12} {'胜率':>10}")
        print("-"*60)
        for _, row in rec_df.iterrows():
            print(f"{row['symbol']:<8} {row['buy_threshold']:>5} {row['interval_days']:>5}d {row['and_threshold']:>5} {row['or_threshold']:>5} {row['total_return']:>+10.1f}% {row['win_rate']:>9.1f}%")

    # 保存结果
    summary_df = pd.DataFrame([{
        'symbol': r['symbol'],
        'total_return': r['total_return'],
        'final_value': r['final_value'],
        'win_rate': r['win_rate']
    } for r in all_results])
    summary_df.to_csv(f'interval_buy_optimized_summary_{timestamp}.csv', index=False)

    if recommendations:
        rec_df.to_csv(f'interval_buy_optimized_params_2026_{timestamp}.csv', index=False)

    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_df['date'] = trades_df['date'].dt.strftime('%Y-%m-%d')
        trades_df.to_csv(f'interval_buy_optimized_trades_{timestamp}.csv', index=False)

    # 保存每个窗口的详细参数
    window_details = []
    for r in all_results:
        for w in r['results']:
            window_details.append({
                'symbol': r['symbol'],
                'window': w['window'],
                'buy_threshold': w['buy_threshold'],
                'interval_days': w['interval_days'],
                'and_threshold': w['and_threshold'],
                'or_threshold': w['or_threshold'],
                'train_return': w['train_return'],
                'test_return': w['test_return'],
            })
    if window_details:
        pd.DataFrame(window_details).to_csv(f'interval_buy_optimized_window_params_{timestamp}.csv', index=False)

    print(f"\n输出文件:")
    print(f"  - interval_buy_optimized_summary_{timestamp}.csv")
    print(f"  - interval_buy_optimized_params_2026_{timestamp}.csv")
    print(f"  - interval_buy_optimized_trades_{timestamp}.csv")
    print(f"  - interval_buy_optimized_window_params_{timestamp}.csv")


if __name__ == "__main__":
    main()
