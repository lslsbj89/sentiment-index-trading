"""
阈值分批买入策略 - MA50斜率增强卖出实验
=====================================
策略: D.阈值分批 + 乘法放宽
数据源: fear_greed_index_s3 (Smoothing=3, MoneyFlow=13)

实验目的:
  测试在AND卖出条件中增加MA50斜率判断是否能减少过早卖出

买入逻辑 (不变):
- 分4档买入，每档25%仓位
- 档位: sentiment < 5/0/-5/-10

卖出逻辑:
- OR条件 (不变): sentiment > OR阈值 → 无条件卖出
- AND条件 (训练期, 不变): sentiment > AND阈值 且 price < MA50 → 卖出
- AND条件 (测试期, 新):  sentiment > AND阈值 且 price < MA50 且 MA50斜率<0 → 卖出
  → 仅当MA50趋势向下时才执行AND卖出，避免在回调中过早卖出

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

BUY_THRESHOLDS = [5, 0, -5, -10]
BATCH_PCT = 0.25

AND_SELL_RANGE = [5, 10, 15, 20, 25]
OR_SELL_RANGE = [30, 40, 50, 55, 60]

# MA50斜率参数
MA50_SLOPE_DAYS = 5  # 用5日差值判断MA50趋势

WINDOWS = [
    {"name": "W2020", "train": ("2016-01-01", "2019-12-31"), "test": ("2020-01-01", "2020-12-31")},
    {"name": "W2021", "train": ("2017-01-01", "2020-12-31"), "test": ("2021-01-01", "2021-12-31")},
    {"name": "W2022", "train": ("2018-01-01", "2021-12-31"), "test": ("2022-01-01", "2022-12-31")},
    {"name": "W2023", "train": ("2019-01-01", "2022-12-31"), "test": ("2023-01-01", "2023-12-31")},
    {"name": "W2024", "train": ("2020-01-01", "2023-12-31"), "test": ("2024-01-01", "2024-12-31")},
    {"name": "W2025", "train": ("2021-01-01", "2024-12-31"), "test": ("2025-01-01", "2025-12-31")},
]

SYMBOLS = ["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META"]


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
    df['MA50_slope'] = df['MA50'] - df['MA50'].shift(MA50_SLOPE_DAYS)
    df = df.dropna()
    df = df[start_date:end_date]
    return df


# ============================================================
# 阈值分批策略核心
# ============================================================

def run_threshold_staged(df, and_threshold, or_threshold,
                         cash, position, entry_price, bought_levels=None,
                         use_ma50_slope=False):
    """
    阈值分批买入策略

    Parameters:
    -----------
    use_ma50_slope : bool
        True: AND卖出需要 MA50斜率<0 (测试期使用)
        False: AND卖出不检查斜率 (训练期使用)
    """
    if bought_levels is None:
        bought_levels = set()

    trades = []
    portfolio_values = []

    initial_capital_for_batch = cash + position * df['Close'].iloc[0] if len(df) > 0 and position > 0 else cash

    for i in range(len(df)):
        current_date = df.index[i]
        current_price = df['Close'].iloc[i]
        current_sentiment = df['sentiment'].iloc[i]
        current_ma50 = df['MA50'].iloc[i]
        current_ma50_slope = df['MA50_slope'].iloc[i]

        # ========== 卖出逻辑 ==========
        if position > 0:
            sell_signal = False
            sell_reason = ""

            # OR条件: 无条件卖出 (不变)
            if current_sentiment > or_threshold:
                sell_signal = True
                sell_reason = f"OR: sentiment {current_sentiment:.1f} > {or_threshold}"
            # AND条件
            elif current_sentiment > and_threshold and current_price < current_ma50:
                if use_ma50_slope:
                    # 测试期: 额外要求MA50斜率为负
                    if current_ma50_slope < 0:
                        sell_signal = True
                        sell_reason = (f"AND+slope: sentiment {current_sentiment:.1f} > {and_threshold}"
                                       f" & price < MA50 & MA50_slope {current_ma50_slope:.2f} < 0")
                    # else: MA50仍向上, 跳过卖出
                else:
                    # 训练期: 原始逻辑
                    sell_signal = True
                    sell_reason = f"AND: sentiment {current_sentiment:.1f} > {and_threshold} & price < MA50"

            if sell_signal:
                sell_price = current_price * (1 - SLIPPAGE) * (1 - COMMISSION)
                profit_pct = (sell_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
                sell_value = position * sell_price
                cash += sell_value

                trades.append({
                    'type': 'SELL',
                    'date': current_date,
                    'price': current_price,
                    'shares': position,
                    'value': sell_value,
                    'sentiment': current_sentiment,
                    'reason': sell_reason,
                    'profit_pct': profit_pct
                })

                position = 0
                entry_price = 0
                bought_levels = set()
                initial_capital_for_batch = cash

        # ========== 买入逻辑: 阈值分批 ==========
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
                        'type': 'BUY',
                        'date': current_date,
                        'price': current_price,
                        'shares': shares,
                        'value': buy_cost,
                        'sentiment': current_sentiment,
                        'reason': f"Level {level_idx+1}/4: sentiment {current_sentiment:.1f} < {threshold}",
                        'batch': level_idx + 1
                    })

        total_value = cash + position * current_price
        portfolio_values.append({
            'date': current_date,
            'value': total_value,
            'cash': cash,
            'position': position,
            'price': current_price
        })

    final_value = cash + position * df['Close'].iloc[-1] if len(df) > 0 else cash
    portfolio_df = pd.DataFrame(portfolio_values).set_index('date') if portfolio_values else pd.DataFrame()

    return final_value, trades, portfolio_df, cash, position, entry_price, bought_levels


# ============================================================
# 网格搜索 (训练期, 不使用斜率条件)
# ============================================================

def grid_search(train_df):
    """网格搜索最优卖出参数 (训练期不使用MA50斜率)"""
    best_return = -float('inf')
    best_params = None

    for and_t in AND_SELL_RANGE:
        for or_t in OR_SELL_RANGE:
            final_value, _, _, _, _, _, _ = run_threshold_staged(
                train_df, and_t, or_t, INITIAL_CAPITAL, 0, 0, None,
                use_ma50_slope=False
            )
            ret = (final_value / INITIAL_CAPITAL - 1) * 100

            if ret > best_return:
                best_return = ret
                best_params = (and_t, or_t)

    return best_params, best_return


# ============================================================
# Walk-Forward 分析
# ============================================================

def run_walk_forward(symbol):
    """对单个股票运行Walk-Forward分析"""
    print(f"\n{'='*70}")
    print(f"  {symbol} - 阈值分批策略 + MA50斜率增强 Walk-Forward 分析")
    print(f"{'='*70}")

    price_df = load_price(symbol)
    sentiment_df = load_sentiment_s3(symbol)

    print(f"  价格数据: {price_df.index.min().date()} ~ {price_df.index.max().date()}")
    print(f"  情绪数据: {sentiment_df.index.min().date()} ~ {sentiment_df.index.max().date()}")
    print(f"  买入档位: {BUY_THRESHOLDS}, 每档{BATCH_PCT*100:.0f}%")
    print(f"  MA50斜率窗口: {MA50_SLOPE_DAYS}日")

    cash = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    bought_levels = None

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

        # 训练期: 网格搜索 (不使用斜率条件)
        best_params, train_return = grid_search(train_df)
        and_t, or_t = best_params

        test_start_value = cash + position * test_df['Close'].iloc[0] if len(test_df) > 0 else cash

        # 测试期: 使用MA50斜率增强条件
        final_value, trades, portfolio_df, cash, position, entry_price, bought_levels = run_threshold_staged(
            test_df, and_t, or_t, cash, position, entry_price, bought_levels,
            use_ma50_slope=True
        )

        test_return = (final_value / test_start_value - 1) * 100 if test_start_value > 0 else 0

        buy_trades = [t for t in trades if t['type'] == 'BUY']
        sell_trades = [t for t in trades if t['type'] == 'SELL']

        results.append({
            'window': window_name,
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

        print(f"\n  {window_name}: Train {train_start[:4]}-{train_end[:4]} → Test {test_start[:4]}")
        print(f"    训练: AND>{and_t}, OR>{or_t} → +{train_return:.1f}%")
        print(f"    测试: {test_return:+.1f}% | 买入{len(buy_trades)}次, 卖出{len(sell_trades)}次")

        if trades:
            for t in trades:
                if t['type'] == 'BUY':
                    print(f"      {t['date'].strftime('%Y-%m-%d')} BUY  ${t['price']:.2f} | {t['reason']}")
                else:
                    print(f"      {t['date'].strftime('%Y-%m-%d')} SELL ${t['price']:.2f} | {t['profit_pct']:+.1f}% | {t['reason']}")

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
    print("阈值分批买入策略 + MA50斜率增强卖出 - Walk-Forward 验证")
    print("="*70)
    print(f"买入档位: {BUY_THRESHOLDS}")
    print(f"每档仓位: {BATCH_PCT*100:.0f}%")
    print(f"初始资金: ${INITIAL_CAPITAL:,}")
    print(f"MA50斜率窗口: {MA50_SLOPE_DAYS}日")
    print(f"\n实验变量:")
    print(f"  训练期AND: sentiment > AND 且 price < MA50 (原始)")
    print(f"  测试期AND: sentiment > AND 且 price < MA50 且 MA50斜率<0 (增强)")

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
                'and_threshold': r['and_threshold'],
                'or_threshold': r['or_threshold'],
                'total_return': result['total_return'],
                'win_rate': result['win_rate']
            })

    # ========== 打印汇总 ==========
    print("\n" + "="*70)
    print("最终排名 (2020-2025 连续回测)")
    print("="*70)

    all_results.sort(key=lambda x: x['total_return'], reverse=True)

    print(f"\n{'排名':<4} {'股票':<8} {'最终资产':>14} {'总收益':>12} {'胜率':>10}")
    print("-"*55)
    for i, r in enumerate(all_results, 1):
        print(f"{i:<4} {r['symbol']:<8} ${r['final_value']:>12,.0f} {r['total_return']:>+10.1f}% {r['win_rate']:>9.1f}%")

    # ========== 2026年推荐参数 ==========
    print("\n" + "="*70)
    print("2026年推荐参数 (基于2021-2024训练)")
    print("="*70)

    rec_df = pd.DataFrame(recommendations)
    rec_df = rec_df.sort_values('total_return', ascending=False)

    print(f"\n{'股票':<8} {'AND阈值':>10} {'OR阈值':>10} {'总收益':>12} {'胜率':>10}")
    print("-"*55)
    for _, row in rec_df.iterrows():
        print(f"{row['symbol']:<8} >{row['and_threshold']:<9} >{row['or_threshold']:<9} {row['total_return']:>+10.1f}% {row['win_rate']:>9.1f}%")

    # ========== 保存结果 ==========
    summary_df = pd.DataFrame([{
        'symbol': r['symbol'],
        'total_return': r['total_return'],
        'final_value': r['final_value'],
        'win_rate': r['win_rate']
    } for r in all_results])
    summary_df.to_csv(f'threshold_staged_ma50slope_summary_{timestamp}.csv', index=False)

    rec_df.to_csv(f'threshold_staged_ma50slope_params_2026_{timestamp}.csv', index=False)

    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_df['date'] = trades_df['date'].dt.strftime('%Y-%m-%d')
        trades_df.to_csv(f'threshold_staged_ma50slope_trades_{timestamp}.csv', index=False)

    print(f"\n输出文件:")
    print(f"  - threshold_staged_ma50slope_summary_{timestamp}.csv")
    print(f"  - threshold_staged_ma50slope_params_2026_{timestamp}.csv")
    print(f"  - threshold_staged_ma50slope_trades_{timestamp}.csv")


if __name__ == "__main__":
    main()
