"""
买入策略对比实验 - Walk-Forward验证 + 加法放宽
使用 fear_greed_index_s3 表 (Smoothing=3)

放宽策略 (加法):
- 买入阈值: +5 (放宽买入条件)
- OR卖出阈值: -5 (提前卖出)
- AND卖出阈值: -3 (提前卖出)

对比5种买入策略:
A. 基准策略: sentiment < threshold → 一次性买入80%
B. MA确认策略: sentiment < threshold AND price > MA10 → 买入
C. 时间分批策略: 首次买入40%, 5天后确认再买40%
D. 阈值分批策略: 多个阈值分批买入 (每档25%)
E. 反转确认策略: 等待sentiment开始回升后买入
"""

import sys
sys.path.insert(0, '/Users/sc2025/Desktop/Claude/Quantrade/src')

import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime
import os
from data_loader import DataLoader

# 数据库配置
db_config = {
    "host": "localhost",
    "port": 5432,
    "database": "crypto_fear_greed_2",
    "user": "sc2025",
    "password": ""
}

# 基本参数
INITIAL_CAPITAL = 100000
COMMISSION = 0.001
SLIPPAGE = 0.001

# 测试股票
TEST_SYMBOL = "NVDA"

# 加法放宽参数
BUY_RELAX_ADD = 5      # 买入阈值 +5
OR_RELAX_SUB = 5       # OR卖出阈值 -5
AND_RELAX_SUB = 3      # AND卖出阈值 -3

# Walk-Forward 窗口
WINDOWS = [
    {"name": "W2020", "train": ("2016-01-01", "2019-12-31"), "test": ("2020-01-01", "2020-12-31")},
    {"name": "W2021", "train": ("2017-01-01", "2020-12-31"), "test": ("2021-01-01", "2021-12-31")},
    {"name": "W2022", "train": ("2018-01-01", "2021-12-31"), "test": ("2022-01-01", "2022-12-31")},
    {"name": "W2023", "train": ("2019-01-01", "2022-12-31"), "test": ("2023-01-01", "2023-12-31")},
    {"name": "W2024", "train": ("2020-01-01", "2023-12-31"), "test": ("2024-01-01", "2024-12-31")},
    {"name": "W2025", "train": ("2021-01-01", "2024-12-31"), "test": ("2025-01-01", "2025-12-31")},
]

# 网格搜索参数范围
BUY_THRESHOLDS = [-20, -15, -10, -5, 0, 5, 10]
AND_SELL_THRESHOLDS = [5, 10, 15, 20, 25]
OR_THRESHOLDS = [30, 40, 50, 55, 60]


def load_sentiment_s3(symbol):
    """从 fear_greed_index_s3 表加载情绪数据"""
    conn = psycopg2.connect(**db_config)
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
    loader = DataLoader(db_config)
    price_df = loader.load_ohlcv(symbol, start_date="2011-01-01")
    loader.close()
    return price_df


def prepare_data(price_df, sentiment_df, start_date, end_date):
    """准备回测数据"""
    df = price_df.copy()
    df['sentiment'] = sentiment_df['smoothed_index'].reindex(df.index)
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['sentiment_prev'] = df['sentiment'].shift(1)
    df = df.dropna()
    df = df[start_date:end_date]
    return df


# ============================================================
# 策略实现 - 支持连续资金和持仓
# ============================================================

def run_strategy_a(df, buy_t, and_t, or_t, cash, position, entry_price, position_pct=0.8):
    """策略A: 基准策略"""
    trades = []
    portfolio_values = []

    for i in range(len(df)):
        current_date = df.index[i]
        current_price = df['Close'].iloc[i]
        current_sentiment = df['sentiment'].iloc[i]
        current_ma50 = df['MA50'].iloc[i]

        # 卖出逻辑
        if position > 0:
            sell_signal = False
            sell_reason = ""
            if current_sentiment > or_t:
                sell_signal = True
                sell_reason = f"OR: {current_sentiment:.1f}>{or_t}"
            elif current_sentiment > and_t and current_price < current_ma50:
                sell_signal = True
                sell_reason = f"AND: {current_sentiment:.1f}>{and_t}"

            if sell_signal:
                sell_price = current_price * (1 - SLIPPAGE) * (1 - COMMISSION)
                profit_pct = (sell_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
                cash += position * sell_price
                trades.append({
                    'type': 'SELL', 'date': current_date, 'price': current_price,
                    'shares': position, 'sentiment': current_sentiment,
                    'reason': sell_reason, 'profit_pct': profit_pct
                })
                position = 0
                entry_price = 0

        # 买入逻辑
        elif position == 0:
            if current_sentiment < buy_t:
                buy_price = current_price * (1 + SLIPPAGE) * (1 + COMMISSION)
                target_value = cash * position_pct
                shares = int(target_value / buy_price)
                if shares > 0 and cash >= shares * buy_price:
                    cash -= shares * buy_price
                    position = shares
                    entry_price = buy_price
                    trades.append({
                        'type': 'BUY', 'date': current_date, 'price': current_price,
                        'shares': shares, 'sentiment': current_sentiment,
                        'reason': f"sent {current_sentiment:.1f}<{buy_t}"
                    })

        total_value = cash + position * current_price
        portfolio_values.append({'date': current_date, 'value': total_value})

    final_value = cash + position * df['Close'].iloc[-1] if len(df) > 0 else cash
    portfolio_df = pd.DataFrame(portfolio_values).set_index('date') if portfolio_values else pd.DataFrame()
    return final_value, trades, portfolio_df, cash, position, entry_price


def run_strategy_b(df, buy_t, and_t, or_t, cash, position, entry_price, position_pct=0.8):
    """策略B: MA确认策略"""
    trades = []
    portfolio_values = []

    for i in range(len(df)):
        current_date = df.index[i]
        current_price = df['Close'].iloc[i]
        current_sentiment = df['sentiment'].iloc[i]
        current_ma10 = df['MA10'].iloc[i]
        current_ma50 = df['MA50'].iloc[i]

        # 卖出逻辑
        if position > 0:
            sell_signal = False
            sell_reason = ""
            if current_sentiment > or_t:
                sell_signal = True
                sell_reason = f"OR: {current_sentiment:.1f}>{or_t}"
            elif current_sentiment > and_t and current_price < current_ma50:
                sell_signal = True
                sell_reason = f"AND: {current_sentiment:.1f}>{and_t}"

            if sell_signal:
                sell_price = current_price * (1 - SLIPPAGE) * (1 - COMMISSION)
                profit_pct = (sell_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
                cash += position * sell_price
                trades.append({
                    'type': 'SELL', 'date': current_date, 'price': current_price,
                    'shares': position, 'sentiment': current_sentiment,
                    'reason': sell_reason, 'profit_pct': profit_pct
                })
                position = 0
                entry_price = 0

        # 买入逻辑: 增加MA10确认
        elif position == 0:
            if current_sentiment < buy_t and current_price > current_ma10:
                buy_price = current_price * (1 + SLIPPAGE) * (1 + COMMISSION)
                target_value = cash * position_pct
                shares = int(target_value / buy_price)
                if shares > 0 and cash >= shares * buy_price:
                    cash -= shares * buy_price
                    position = shares
                    entry_price = buy_price
                    trades.append({
                        'type': 'BUY', 'date': current_date, 'price': current_price,
                        'shares': shares, 'sentiment': current_sentiment,
                        'reason': f"sent {current_sentiment:.1f}<{buy_t} & P>MA10"
                    })

        total_value = cash + position * current_price
        portfolio_values.append({'date': current_date, 'value': total_value})

    final_value = cash + position * df['Close'].iloc[-1] if len(df) > 0 else cash
    portfolio_df = pd.DataFrame(portfolio_values).set_index('date') if portfolio_values else pd.DataFrame()
    return final_value, trades, portfolio_df, cash, position, entry_price


def run_strategy_c(df, buy_t, and_t, or_t, cash, position, entry_price,
                   first_buy_date=None, bought_second=False):
    """策略C: 时间分批策略"""
    trades = []
    portfolio_values = []

    for i in range(len(df)):
        current_date = df.index[i]
        current_price = df['Close'].iloc[i]
        current_sentiment = df['sentiment'].iloc[i]
        current_ma10 = df['MA10'].iloc[i]
        current_ma50 = df['MA50'].iloc[i]

        # 卖出逻辑
        if position > 0:
            sell_signal = False
            sell_reason = ""
            if current_sentiment > or_t:
                sell_signal = True
                sell_reason = f"OR: {current_sentiment:.1f}>{or_t}"
            elif current_sentiment > and_t and current_price < current_ma50:
                sell_signal = True
                sell_reason = f"AND: {current_sentiment:.1f}>{and_t}"

            if sell_signal:
                sell_price = current_price * (1 - SLIPPAGE) * (1 - COMMISSION)
                profit_pct = (sell_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
                cash += position * sell_price
                trades.append({
                    'type': 'SELL', 'date': current_date, 'price': current_price,
                    'shares': position, 'sentiment': current_sentiment,
                    'reason': sell_reason, 'profit_pct': profit_pct
                })
                position = 0
                entry_price = 0
                first_buy_date = None
                bought_second = False

        # 买入逻辑: 分批
        if position == 0 and first_buy_date is None:
            if current_sentiment < buy_t:
                buy_price = current_price * (1 + SLIPPAGE) * (1 + COMMISSION)
                target_value = cash * 0.4
                shares = int(target_value / buy_price)
                if shares > 0 and cash >= shares * buy_price:
                    cash -= shares * buy_price
                    position = shares
                    entry_price = buy_price
                    first_buy_date = current_date
                    bought_second = False
                    trades.append({
                        'type': 'BUY', 'date': current_date, 'price': current_price,
                        'shares': shares, 'sentiment': current_sentiment,
                        'reason': f"Batch1(40%): sent {current_sentiment:.1f}<{buy_t}"
                    })

        elif position > 0 and not bought_second and first_buy_date is not None:
            days_held = (current_date - first_buy_date).days
            if days_held >= 5 and current_price > current_ma10:
                buy_price = current_price * (1 + SLIPPAGE) * (1 + COMMISSION)
                target_value = cash * 0.8
                shares = int(target_value / buy_price)
                if shares > 0 and cash >= shares * buy_price:
                    total_cost = entry_price * position + buy_price * shares
                    position += shares
                    entry_price = total_cost / position
                    cash -= shares * buy_price
                    bought_second = True
                    trades.append({
                        'type': 'BUY', 'date': current_date, 'price': current_price,
                        'shares': shares, 'sentiment': current_sentiment,
                        'reason': f"Batch2: {days_held}d & P>MA10"
                    })

        total_value = cash + position * current_price
        portfolio_values.append({'date': current_date, 'value': total_value})

    final_value = cash + position * df['Close'].iloc[-1] if len(df) > 0 else cash
    portfolio_df = pd.DataFrame(portfolio_values).set_index('date') if portfolio_values else pd.DataFrame()
    return final_value, trades, portfolio_df, cash, position, entry_price, first_buy_date, bought_second


def run_strategy_d(df, and_t, or_t, cash, position, entry_price, bought_levels=None):
    """策略D: 阈值分批策略"""
    THRESHOLDS = [10, 5, 0, -5]  # 放宽后的阈值档位
    BATCH_PCT = 0.25

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

        # 卖出逻辑
        if position > 0:
            sell_signal = False
            sell_reason = ""
            if current_sentiment > or_t:
                sell_signal = True
                sell_reason = f"OR: {current_sentiment:.1f}>{or_t}"
            elif current_sentiment > and_t and current_price < current_ma50:
                sell_signal = True
                sell_reason = f"AND: {current_sentiment:.1f}>{and_t}"

            if sell_signal:
                sell_price = current_price * (1 - SLIPPAGE) * (1 - COMMISSION)
                profit_pct = (sell_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
                cash += position * sell_price
                trades.append({
                    'type': 'SELL', 'date': current_date, 'price': current_price,
                    'shares': position, 'sentiment': current_sentiment,
                    'reason': sell_reason, 'profit_pct': profit_pct
                })
                position = 0
                entry_price = 0
                bought_levels = set()
                initial_capital_for_batch = cash

        # 买入逻辑: 按阈值分批
        for level_idx, threshold in enumerate(THRESHOLDS):
            if level_idx not in bought_levels and current_sentiment < threshold:
                buy_price = current_price * (1 + SLIPPAGE) * (1 + COMMISSION)
                target_value = initial_capital_for_batch * BATCH_PCT
                shares = int(target_value / buy_price)
                if shares > 0 and cash >= shares * buy_price:
                    if position > 0:
                        total_cost = entry_price * position + buy_price * shares
                        position += shares
                        entry_price = total_cost / position
                    else:
                        position = shares
                        entry_price = buy_price
                    cash -= shares * buy_price
                    bought_levels.add(level_idx)
                    trades.append({
                        'type': 'BUY', 'date': current_date, 'price': current_price,
                        'shares': shares, 'sentiment': current_sentiment,
                        'reason': f"Batch{level_idx+1}(25%): sent {current_sentiment:.1f}<{threshold}"
                    })

        total_value = cash + position * current_price
        portfolio_values.append({'date': current_date, 'value': total_value})

    final_value = cash + position * df['Close'].iloc[-1] if len(df) > 0 else cash
    portfolio_df = pd.DataFrame(portfolio_values).set_index('date') if portfolio_values else pd.DataFrame()
    return final_value, trades, portfolio_df, cash, position, entry_price, bought_levels


def run_strategy_e(df, buy_t, and_t, or_t, cash, position, entry_price,
                   in_watch_mode=False, position_pct=0.8):
    """策略E: 反转确认策略"""
    trades = []
    portfolio_values = []

    for i in range(len(df)):
        current_date = df.index[i]
        current_price = df['Close'].iloc[i]
        current_sentiment = df['sentiment'].iloc[i]
        prev_sentiment = df['sentiment_prev'].iloc[i]
        current_ma50 = df['MA50'].iloc[i]

        # 卖出逻辑
        if position > 0:
            sell_signal = False
            sell_reason = ""
            if current_sentiment > or_t:
                sell_signal = True
                sell_reason = f"OR: {current_sentiment:.1f}>{or_t}"
            elif current_sentiment > and_t and current_price < current_ma50:
                sell_signal = True
                sell_reason = f"AND: {current_sentiment:.1f}>{and_t}"

            if sell_signal:
                sell_price = current_price * (1 - SLIPPAGE) * (1 - COMMISSION)
                profit_pct = (sell_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
                cash += position * sell_price
                trades.append({
                    'type': 'SELL', 'date': current_date, 'price': current_price,
                    'shares': position, 'sentiment': current_sentiment,
                    'reason': sell_reason, 'profit_pct': profit_pct
                })
                position = 0
                entry_price = 0
                in_watch_mode = False

        # 买入逻辑: 两阶段
        elif position == 0:
            if not in_watch_mode and current_sentiment < buy_t:
                in_watch_mode = True

            if in_watch_mode and current_sentiment > prev_sentiment and current_sentiment < buy_t + 10:
                buy_price = current_price * (1 + SLIPPAGE) * (1 + COMMISSION)
                target_value = cash * position_pct
                shares = int(target_value / buy_price)
                if shares > 0 and cash >= shares * buy_price:
                    cash -= shares * buy_price
                    position = shares
                    entry_price = buy_price
                    in_watch_mode = False
                    trades.append({
                        'type': 'BUY', 'date': current_date, 'price': current_price,
                        'shares': shares, 'sentiment': current_sentiment,
                        'reason': f"Reversal: {prev_sentiment:.1f}->{current_sentiment:.1f}"
                    })

            if in_watch_mode and current_sentiment > buy_t + 15:
                in_watch_mode = False

        total_value = cash + position * current_price
        portfolio_values.append({'date': current_date, 'value': total_value})

    final_value = cash + position * df['Close'].iloc[-1] if len(df) > 0 else cash
    portfolio_df = pd.DataFrame(portfolio_values).set_index('date') if portfolio_values else pd.DataFrame()
    return final_value, trades, portfolio_df, cash, position, entry_price, in_watch_mode


# ============================================================
# 网格搜索
# ============================================================

def grid_search_strategy_a(df, position_pct=0.8):
    """策略A网格搜索"""
    best_return = -float('inf')
    best_params = None

    for buy_t in BUY_THRESHOLDS:
        for and_t in AND_SELL_THRESHOLDS:
            for or_t in OR_THRESHOLDS:
                final_value, _, _, _, _, _ = run_strategy_a(
                    df, buy_t, and_t, or_t, INITIAL_CAPITAL, 0, 0, position_pct
                )
                ret = (final_value / INITIAL_CAPITAL - 1) * 100
                if ret > best_return:
                    best_return = ret
                    best_params = (buy_t, and_t, or_t)

    return best_params, best_return


def grid_search_strategy_b(df, position_pct=0.8):
    """策略B网格搜索"""
    best_return = -float('inf')
    best_params = None

    for buy_t in BUY_THRESHOLDS:
        for and_t in AND_SELL_THRESHOLDS:
            for or_t in OR_THRESHOLDS:
                final_value, _, _, _, _, _ = run_strategy_b(
                    df, buy_t, and_t, or_t, INITIAL_CAPITAL, 0, 0, position_pct
                )
                ret = (final_value / INITIAL_CAPITAL - 1) * 100
                if ret > best_return:
                    best_return = ret
                    best_params = (buy_t, and_t, or_t)

    return best_params, best_return


def grid_search_strategy_c(df):
    """策略C网格搜索"""
    best_return = -float('inf')
    best_params = None

    for buy_t in BUY_THRESHOLDS:
        for and_t in AND_SELL_THRESHOLDS:
            for or_t in OR_THRESHOLDS:
                final_value, _, _, _, _, _, _, _ = run_strategy_c(
                    df, buy_t, and_t, or_t, INITIAL_CAPITAL, 0, 0, None, False
                )
                ret = (final_value / INITIAL_CAPITAL - 1) * 100
                if ret > best_return:
                    best_return = ret
                    best_params = (buy_t, and_t, or_t)

    return best_params, best_return


def grid_search_strategy_d(df):
    """策略D网格搜索 (只搜索卖出参数)"""
    best_return = -float('inf')
    best_params = None

    for and_t in AND_SELL_THRESHOLDS:
        for or_t in OR_THRESHOLDS:
            final_value, _, _, _, _, _, _ = run_strategy_d(
                df, and_t, or_t, INITIAL_CAPITAL, 0, 0, None
            )
            ret = (final_value / INITIAL_CAPITAL - 1) * 100
            if ret > best_return:
                best_return = ret
                best_params = (and_t, or_t)

    return best_params, best_return


def grid_search_strategy_e(df, position_pct=0.8):
    """策略E网格搜索"""
    best_return = -float('inf')
    best_params = None

    for buy_t in BUY_THRESHOLDS:
        for and_t in AND_SELL_THRESHOLDS:
            for or_t in OR_THRESHOLDS:
                final_value, _, _, _, _, _, _ = run_strategy_e(
                    df, buy_t, and_t, or_t, INITIAL_CAPITAL, 0, 0, False, position_pct
                )
                ret = (final_value / INITIAL_CAPITAL - 1) * 100
                if ret > best_return:
                    best_return = ret
                    best_params = (buy_t, and_t, or_t)

    return best_params, best_return


# ============================================================
# Walk-Forward 主流程
# ============================================================

def run_walk_forward(symbol):
    """运行Walk-Forward分析"""
    print(f"\n加载 {symbol} 数据...")
    price_df = load_price(symbol)
    sentiment_df = load_sentiment_s3(symbol)
    print(f"  价格数据: {price_df.index.min().date()} ~ {price_df.index.max().date()}")
    print(f"  情绪数据: {sentiment_df.index.min().date()} ~ {sentiment_df.index.max().date()}")

    # 各策略的连续状态
    strategies = {
        'A.Baseline': {'cash': INITIAL_CAPITAL, 'position': 0, 'entry_price': 0, 'extra': {}},
        'B.MA_Confirm': {'cash': INITIAL_CAPITAL, 'position': 0, 'entry_price': 0, 'extra': {}},
        'C.Time_Staged': {'cash': INITIAL_CAPITAL, 'position': 0, 'entry_price': 0,
                         'extra': {'first_buy_date': None, 'bought_second': False}},
        'D.Threshold_Staged': {'cash': INITIAL_CAPITAL, 'position': 0, 'entry_price': 0,
                              'extra': {'bought_levels': None}},
        'E.Reversal_Confirm': {'cash': INITIAL_CAPITAL, 'position': 0, 'entry_price': 0,
                              'extra': {'in_watch_mode': False}},
    }

    all_results = {name: [] for name in strategies.keys()}
    all_trades = {name: [] for name in strategies.keys()}

    for window in WINDOWS:
        window_name = window['name']
        train_start, train_end = window['train']
        test_start, test_end = window['test']

        print(f"\n{'='*60}")
        print(f"  {window_name}: Train {train_start[:4]}-{train_end[:4]} → Test {test_start[:4]}")
        print(f"{'='*60}")

        # 准备训练和测试数据
        train_df = prepare_data(price_df, sentiment_df, train_start, train_end)
        test_df = prepare_data(price_df, sentiment_df, test_start, test_end)

        if len(train_df) < 100 or len(test_df) < 10:
            print(f"  数据不足，跳过")
            continue

        # ========== 策略A ==========
        print(f"\n  A.Baseline:")
        params_a, train_ret_a = grid_search_strategy_a(train_df)
        buy_t, and_t, or_t = params_a
        # 加法放宽
        test_buy_t = buy_t + BUY_RELAX_ADD
        test_and_t = and_t - AND_RELAX_SUB
        test_or_t = or_t - OR_RELAX_SUB

        state = strategies['A.Baseline']
        start_value = state['cash'] + state['position'] * test_df['Close'].iloc[0]

        final_value, trades, portfolio_df, cash, position, entry_price = run_strategy_a(
            test_df, test_buy_t, test_and_t, test_or_t, state['cash'], state['position'], state['entry_price']
        )

        test_ret = (final_value / start_value - 1) * 100 if start_value > 0 else 0
        strategies['A.Baseline'].update({'cash': cash, 'position': position, 'entry_price': entry_price})

        print(f"    Train: Buy<{buy_t}, AND>{and_t}, OR>{or_t} → +{train_ret_a:.1f}%")
        print(f"    Test:  Buy<{test_buy_t}, AND>{test_and_t}, OR>{test_or_t} → {test_ret:+.1f}% | Trades: {len([t for t in trades if t['type']=='BUY'])}")

        all_results['A.Baseline'].append({
            'window': window_name, 'params': params_a, 'train_ret': train_ret_a,
            'test_ret': test_ret, 'end_value': final_value
        })
        for t in trades:
            t['window'] = window_name
        all_trades['A.Baseline'].extend(trades)

        # ========== 策略B ==========
        print(f"\n  B.MA_Confirm:")
        params_b, train_ret_b = grid_search_strategy_b(train_df)
        buy_t, and_t, or_t = params_b
        test_buy_t = buy_t + BUY_RELAX_ADD
        test_and_t = and_t - AND_RELAX_SUB
        test_or_t = or_t - OR_RELAX_SUB

        state = strategies['B.MA_Confirm']
        start_value = state['cash'] + state['position'] * test_df['Close'].iloc[0]

        final_value, trades, portfolio_df, cash, position, entry_price = run_strategy_b(
            test_df, test_buy_t, test_and_t, test_or_t, state['cash'], state['position'], state['entry_price']
        )

        test_ret = (final_value / start_value - 1) * 100 if start_value > 0 else 0
        strategies['B.MA_Confirm'].update({'cash': cash, 'position': position, 'entry_price': entry_price})

        print(f"    Train: Buy<{buy_t}, AND>{and_t}, OR>{or_t} → +{train_ret_b:.1f}%")
        print(f"    Test:  Buy<{test_buy_t} & P>MA10, AND>{test_and_t}, OR>{test_or_t} → {test_ret:+.1f}% | Trades: {len([t for t in trades if t['type']=='BUY'])}")

        all_results['B.MA_Confirm'].append({
            'window': window_name, 'params': params_b, 'train_ret': train_ret_b,
            'test_ret': test_ret, 'end_value': final_value
        })
        for t in trades:
            t['window'] = window_name
        all_trades['B.MA_Confirm'].extend(trades)

        # ========== 策略C ==========
        print(f"\n  C.Time_Staged:")
        params_c, train_ret_c = grid_search_strategy_c(train_df)
        buy_t, and_t, or_t = params_c
        test_buy_t = buy_t + BUY_RELAX_ADD
        test_and_t = and_t - AND_RELAX_SUB
        test_or_t = or_t - OR_RELAX_SUB

        state = strategies['C.Time_Staged']
        start_value = state['cash'] + state['position'] * test_df['Close'].iloc[0]

        final_value, trades, portfolio_df, cash, position, entry_price, first_buy_date, bought_second = run_strategy_c(
            test_df, test_buy_t, test_and_t, test_or_t, state['cash'], state['position'], state['entry_price'],
            state['extra'].get('first_buy_date'), state['extra'].get('bought_second', False)
        )

        test_ret = (final_value / start_value - 1) * 100 if start_value > 0 else 0
        strategies['C.Time_Staged'].update({
            'cash': cash, 'position': position, 'entry_price': entry_price,
            'extra': {'first_buy_date': first_buy_date, 'bought_second': bought_second}
        })

        print(f"    Train: Buy<{buy_t}, AND>{and_t}, OR>{or_t} → +{train_ret_c:.1f}%")
        print(f"    Test:  Buy<{test_buy_t} (40%+40%), AND>{test_and_t}, OR>{test_or_t} → {test_ret:+.1f}% | Trades: {len([t for t in trades if t['type']=='BUY'])}")

        all_results['C.Time_Staged'].append({
            'window': window_name, 'params': params_c, 'train_ret': train_ret_c,
            'test_ret': test_ret, 'end_value': final_value
        })
        for t in trades:
            t['window'] = window_name
        all_trades['C.Time_Staged'].extend(trades)

        # ========== 策略D ==========
        print(f"\n  D.Threshold_Staged:")
        params_d, train_ret_d = grid_search_strategy_d(train_df)
        and_t, or_t = params_d
        test_and_t = and_t - AND_RELAX_SUB
        test_or_t = or_t - OR_RELAX_SUB

        state = strategies['D.Threshold_Staged']
        start_value = state['cash'] + state['position'] * test_df['Close'].iloc[0]

        final_value, trades, portfolio_df, cash, position, entry_price, bought_levels = run_strategy_d(
            test_df, test_and_t, test_or_t, state['cash'], state['position'], state['entry_price'],
            state['extra'].get('bought_levels')
        )

        test_ret = (final_value / start_value - 1) * 100 if start_value > 0 else 0
        strategies['D.Threshold_Staged'].update({
            'cash': cash, 'position': position, 'entry_price': entry_price,
            'extra': {'bought_levels': bought_levels}
        })

        print(f"    Train: AND>{and_t}, OR>{or_t} → +{train_ret_d:.1f}%")
        print(f"    Test:  4-level (10,5,0,-5), AND>{test_and_t}, OR>{test_or_t} → {test_ret:+.1f}% | Trades: {len([t for t in trades if t['type']=='BUY'])}")

        all_results['D.Threshold_Staged'].append({
            'window': window_name, 'params': params_d, 'train_ret': train_ret_d,
            'test_ret': test_ret, 'end_value': final_value
        })
        for t in trades:
            t['window'] = window_name
        all_trades['D.Threshold_Staged'].extend(trades)

        # ========== 策略E ==========
        print(f"\n  E.Reversal_Confirm:")
        params_e, train_ret_e = grid_search_strategy_e(train_df)
        buy_t, and_t, or_t = params_e
        test_buy_t = buy_t + BUY_RELAX_ADD
        test_and_t = and_t - AND_RELAX_SUB
        test_or_t = or_t - OR_RELAX_SUB

        state = strategies['E.Reversal_Confirm']
        start_value = state['cash'] + state['position'] * test_df['Close'].iloc[0]

        final_value, trades, portfolio_df, cash, position, entry_price, in_watch_mode = run_strategy_e(
            test_df, test_buy_t, test_and_t, test_or_t, state['cash'], state['position'], state['entry_price'],
            state['extra'].get('in_watch_mode', False)
        )

        test_ret = (final_value / start_value - 1) * 100 if start_value > 0 else 0
        strategies['E.Reversal_Confirm'].update({
            'cash': cash, 'position': position, 'entry_price': entry_price,
            'extra': {'in_watch_mode': in_watch_mode}
        })

        print(f"    Train: Buy<{buy_t}, AND>{and_t}, OR>{or_t} → +{train_ret_e:.1f}%")
        print(f"    Test:  Buy<{test_buy_t} (reversal), AND>{test_and_t}, OR>{test_or_t} → {test_ret:+.1f}% | Trades: {len([t for t in trades if t['type']=='BUY'])}")

        all_results['E.Reversal_Confirm'].append({
            'window': window_name, 'params': params_e, 'train_ret': train_ret_e,
            'test_ret': test_ret, 'end_value': final_value
        })
        for t in trades:
            t['window'] = window_name
        all_trades['E.Reversal_Confirm'].extend(trades)

    return strategies, all_results, all_trades


def main():
    """主函数"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("="*70)
    print("买入策略对比实验 - Walk-Forward验证 + 加法放宽")
    print(f"股票: {TEST_SYMBOL}")
    print(f"放宽策略: Buy+{BUY_RELAX_ADD}, OR-{OR_RELAX_SUB}, AND-{AND_RELAX_SUB}")
    print("="*70)

    # 运行Walk-Forward
    strategies, all_results, all_trades = run_walk_forward(TEST_SYMBOL)

    # 打印最终结果
    print("\n" + "="*70)
    print("Walk-Forward 最终结果对比 (加法放宽)")
    print("="*70)

    print(f"\n{'策略':<22} {'最终资产':>14} {'总收益':>12} {'买入次数':>10} {'卖出次数':>10} {'胜率':>10}")
    print("-"*80)

    final_values = {}
    for name, state in strategies.items():
        if all_results[name]:
            final_value = all_results[name][-1]['end_value']
        else:
            final_value = state['cash']

        final_values[name] = final_value
        total_return = (final_value / INITIAL_CAPITAL - 1) * 100

        buy_count = len([t for t in all_trades[name] if t['type'] == 'BUY'])
        sell_count = len([t for t in all_trades[name] if t['type'] == 'SELL'])

        sell_trades = [t for t in all_trades[name] if t['type'] == 'SELL']
        if sell_trades:
            win_rate = len([t for t in sell_trades if t['profit_pct'] > 0]) / len(sell_trades) * 100
        else:
            win_rate = 0

        print(f"{name:<22} ${final_value:>12,.0f} {total_return:>+10.1f}% {buy_count:>10} {sell_count:>10} {win_rate:>9.1f}%")

    # 各窗口详细结果
    print("\n" + "="*70)
    print("各窗口测试收益对比")
    print("="*70)

    print(f"\n{'Window':<8}", end="")
    for name in strategies.keys():
        short_name = name.split('.')[1][:10]
        print(f"{short_name:>12}", end="")
    print()
    print("-"*80)

    for i, window in enumerate(WINDOWS):
        print(f"{window['name']:<8}", end="")
        for name in strategies.keys():
            if i < len(all_results[name]):
                test_ret = all_results[name][i]['test_ret']
                print(f"{test_ret:>+11.1f}%", end="")
            else:
                print(f"{'N/A':>12}", end="")
        print()

    # 保存结果
    summary_data = []
    for name in strategies.keys():
        final_value = final_values[name]
        total_return = (final_value / INITIAL_CAPITAL - 1) * 100
        buy_count = len([t for t in all_trades[name] if t['type'] == 'BUY'])
        sell_count = len([t for t in all_trades[name] if t['type'] == 'SELL'])
        sell_trades = [t for t in all_trades[name] if t['type'] == 'SELL']
        win_rate = len([t for t in sell_trades if t['profit_pct'] > 0]) / len(sell_trades) * 100 if sell_trades else 0

        summary_data.append({
            'strategy': name,
            'final_value': final_value,
            'total_return': total_return,
            'num_buys': buy_count,
            'num_sells': sell_count,
            'win_rate': win_rate
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('total_return', ascending=False)
    summary_df.to_csv(f'wf_additive_comparison_{TEST_SYMBOL}_{timestamp}.csv', index=False)

    # 交易记录
    all_trades_list = []
    for name, trades in all_trades.items():
        for t in trades:
            t_copy = t.copy()
            t_copy['strategy'] = name
            t_copy['date'] = t_copy['date'].strftime('%Y-%m-%d') if hasattr(t_copy['date'], 'strftime') else t_copy['date']
            all_trades_list.append(t_copy)

    if all_trades_list:
        trades_df = pd.DataFrame(all_trades_list)
        trades_df.to_csv(f'wf_additive_trades_{TEST_SYMBOL}_{timestamp}.csv', index=False)

    print(f"\n输出文件:")
    print(f"  - wf_additive_comparison_{TEST_SYMBOL}_{timestamp}.csv")
    print(f"  - wf_additive_trades_{TEST_SYMBOL}_{timestamp}.csv")

    # 结论
    print("\n" + "="*70)
    print("分析结论")
    print("="*70)

    best_strategy = max(final_values.items(), key=lambda x: x[1])
    best_return = (best_strategy[1] / INITIAL_CAPITAL - 1) * 100
    print(f"\n  Walk-Forward + 加法放宽 最佳策略: {best_strategy[0]}")
    print(f"  总收益: {best_return:+.1f}%")
    print(f"  最终资产: ${best_strategy[1]:,.0f}")


if __name__ == "__main__":
    main()
