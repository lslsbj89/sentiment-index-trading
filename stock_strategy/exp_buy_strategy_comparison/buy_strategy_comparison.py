"""
买入策略对比实验
使用 fear_greed_index_s3 表 (Smoothing=3)

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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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

# 测试期间 (使用较长期间以便观察)
TEST_START = "2020-01-01"
TEST_END = "2025-12-31"

# 固定参数 (来自NVDA的最优参数)
BUY_THRESHOLD = 5       # 买入阈值
AND_SELL_THRESHOLD = 22  # AND卖出阈值
OR_SELL_THRESHOLD = 55   # OR卖出阈值


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


def prepare_data(symbol, start_date, end_date):
    """准备回测数据"""
    price_df = load_price(symbol)
    sentiment_df = load_sentiment_s3(symbol)

    df = price_df.copy()
    df['sentiment'] = sentiment_df['smoothed_index'].reindex(df.index)
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['sentiment_prev'] = df['sentiment'].shift(1)
    df = df.dropna()

    # 过滤日期
    df = df[start_date:end_date]
    return df


# ============================================================
# 策略A: 基准策略 (一次性买入80%)
# ============================================================
def strategy_a_baseline(df, buy_threshold, and_threshold, or_threshold, position_pct=0.8):
    """基准策略: sentiment < threshold → 一次性买入"""
    cash = INITIAL_CAPITAL
    position = 0
    entry_price = 0
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

            if current_sentiment > or_threshold:
                sell_signal = True
                sell_reason = f"OR: {current_sentiment:.1f} > {or_threshold}"
            elif current_sentiment > and_threshold and current_price < current_ma50:
                sell_signal = True
                sell_reason = f"AND: {current_sentiment:.1f} > {and_threshold}"

            if sell_signal:
                sell_price = current_price * (1 - SLIPPAGE) * (1 - COMMISSION)
                profit_pct = (sell_price - entry_price) / entry_price * 100
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
            if current_sentiment < buy_threshold:
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
                        'reason': f"sentiment {current_sentiment:.1f} < {buy_threshold}"
                    })

        # 记录组合价值
        total_value = cash + position * current_price
        portfolio_values.append({'date': current_date, 'value': total_value})

    final_value = cash + position * df['Close'].iloc[-1]
    return final_value, trades, pd.DataFrame(portfolio_values).set_index('date')


# ============================================================
# 策略B: MA确认策略
# ============================================================
def strategy_b_ma_confirm(df, buy_threshold, and_threshold, or_threshold, position_pct=0.8):
    """MA确认策略: sentiment < threshold AND price > MA10 → 买入"""
    cash = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    trades = []
    portfolio_values = []

    for i in range(len(df)):
        current_date = df.index[i]
        current_price = df['Close'].iloc[i]
        current_sentiment = df['sentiment'].iloc[i]
        current_ma10 = df['MA10'].iloc[i]
        current_ma50 = df['MA50'].iloc[i]

        # 卖出逻辑 (与基准相同)
        if position > 0:
            sell_signal = False
            sell_reason = ""

            if current_sentiment > or_threshold:
                sell_signal = True
                sell_reason = f"OR: {current_sentiment:.1f} > {or_threshold}"
            elif current_sentiment > and_threshold and current_price < current_ma50:
                sell_signal = True
                sell_reason = f"AND: {current_sentiment:.1f} > {and_threshold}"

            if sell_signal:
                sell_price = current_price * (1 - SLIPPAGE) * (1 - COMMISSION)
                profit_pct = (sell_price - entry_price) / entry_price * 100
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
            if current_sentiment < buy_threshold and current_price > current_ma10:
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
                        'reason': f"sentiment {current_sentiment:.1f} < {buy_threshold} & price > MA10"
                    })

        total_value = cash + position * current_price
        portfolio_values.append({'date': current_date, 'value': total_value})

    final_value = cash + position * df['Close'].iloc[-1]
    return final_value, trades, pd.DataFrame(portfolio_values).set_index('date')


# ============================================================
# 策略C: 时间分批策略
# ============================================================
def strategy_c_staged_time(df, buy_threshold, and_threshold, or_threshold):
    """时间分批策略: 首次买入40%, 5天后如仍持有且价格确认再买40%"""
    cash = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    first_buy_date = None
    bought_second = False
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

            if current_sentiment > or_threshold:
                sell_signal = True
                sell_reason = f"OR: {current_sentiment:.1f} > {or_threshold}"
            elif current_sentiment > and_threshold and current_price < current_ma50:
                sell_signal = True
                sell_reason = f"AND: {current_sentiment:.1f} > {and_threshold}"

            if sell_signal:
                sell_price = current_price * (1 - SLIPPAGE) * (1 - COMMISSION)
                profit_pct = (sell_price - entry_price) / entry_price * 100
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
            # 第一批: 40%
            if current_sentiment < buy_threshold:
                buy_price = current_price * (1 + SLIPPAGE) * (1 + COMMISSION)
                target_value = cash * 0.4  # 第一批40%
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
                        'reason': f"第1批(40%): sentiment {current_sentiment:.1f} < {buy_threshold}"
                    })

        elif position > 0 and not bought_second and first_buy_date is not None:
            # 第二批: 等5天且价格>MA10
            days_held = (current_date - first_buy_date).days
            if days_held >= 5 and current_price > current_ma10:
                buy_price = current_price * (1 + SLIPPAGE) * (1 + COMMISSION)
                target_value = cash * 0.8  # 剩余资金的大部分
                shares = int(target_value / buy_price)

                if shares > 0 and cash >= shares * buy_price:
                    # 更新平均入场价
                    total_cost = entry_price * position + buy_price * shares
                    position += shares
                    entry_price = total_cost / position
                    cash -= shares * buy_price
                    bought_second = True
                    trades.append({
                        'type': 'BUY', 'date': current_date, 'price': current_price,
                        'shares': shares, 'sentiment': current_sentiment,
                        'reason': f"第2批(加仓): 持有{days_held}天 & price > MA10"
                    })

        total_value = cash + position * current_price
        portfolio_values.append({'date': current_date, 'value': total_value})

    final_value = cash + position * df['Close'].iloc[-1]
    return final_value, trades, pd.DataFrame(portfolio_values).set_index('date')


# ============================================================
# 策略D: 阈值分批策略
# ============================================================
def strategy_d_staged_threshold(df, and_threshold, or_threshold):
    """阈值分批策略: 多个阈值分批买入"""
    # 分批阈值
    THRESHOLDS = [5, 0, -5, -10]  # 每档买入25%
    BATCH_PCT = 0.25

    cash = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    bought_levels = set()  # 已买入的阈值级别
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

            if current_sentiment > or_threshold:
                sell_signal = True
                sell_reason = f"OR: {current_sentiment:.1f} > {or_threshold}"
            elif current_sentiment > and_threshold and current_price < current_ma50:
                sell_signal = True
                sell_reason = f"AND: {current_sentiment:.1f} > {and_threshold}"

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
                bought_levels = set()  # 重置

        # 买入逻辑: 按阈值分批
        for level_idx, threshold in enumerate(THRESHOLDS):
            if level_idx not in bought_levels and current_sentiment < threshold:
                buy_price = current_price * (1 + SLIPPAGE) * (1 + COMMISSION)
                target_value = INITIAL_CAPITAL * BATCH_PCT  # 每批固定为初始资金的25%
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
                        'reason': f"第{level_idx+1}批(25%): sentiment {current_sentiment:.1f} < {threshold}"
                    })

        total_value = cash + position * current_price
        portfolio_values.append({'date': current_date, 'value': total_value})

    final_value = cash + position * df['Close'].iloc[-1]
    return final_value, trades, pd.DataFrame(portfolio_values).set_index('date')


# ============================================================
# 策略E: 反转确认策略
# ============================================================
def strategy_e_reversal_confirm(df, buy_threshold, and_threshold, or_threshold, position_pct=0.8):
    """反转确认策略: 等待sentiment开始回升后买入"""
    cash = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    in_watch_mode = False  # 是否进入观察模式
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

            if current_sentiment > or_threshold:
                sell_signal = True
                sell_reason = f"OR: {current_sentiment:.1f} > {or_threshold}"
            elif current_sentiment > and_threshold and current_price < current_ma50:
                sell_signal = True
                sell_reason = f"AND: {current_sentiment:.1f} > {and_threshold}"

            if sell_signal:
                sell_price = current_price * (1 - SLIPPAGE) * (1 - COMMISSION)
                profit_pct = (sell_price - entry_price) / entry_price * 100
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
            # 阶段1: 进入观察模式
            if not in_watch_mode and current_sentiment < buy_threshold:
                in_watch_mode = True

            # 阶段2: 确认反转 (sentiment开始回升)
            if in_watch_mode and current_sentiment > prev_sentiment and current_sentiment < buy_threshold + 10:
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
                        'reason': f"反转确认: sentiment从{prev_sentiment:.1f}回升到{current_sentiment:.1f}"
                    })

            # 如果sentiment回升太多，退出观察模式
            if in_watch_mode and current_sentiment > buy_threshold + 15:
                in_watch_mode = False

        total_value = cash + position * current_price
        portfolio_values.append({'date': current_date, 'value': total_value})

    final_value = cash + position * df['Close'].iloc[-1]
    return final_value, trades, pd.DataFrame(portfolio_values).set_index('date')


def calculate_metrics(portfolio_df, trades):
    """计算策略指标"""
    returns = portfolio_df['value'].pct_change().dropna()

    # 总收益
    total_return = (portfolio_df['value'].iloc[-1] / INITIAL_CAPITAL - 1) * 100

    # 年化收益
    days = (portfolio_df.index[-1] - portfolio_df.index[0]).days
    annual_return = ((1 + total_return/100) ** (365/days) - 1) * 100 if days > 0 else 0

    # 最大回撤
    rolling_max = portfolio_df['value'].cummax()
    drawdown = (portfolio_df['value'] - rolling_max) / rolling_max * 100
    max_drawdown = drawdown.min()

    # 夏普比率
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    # 交易统计
    buy_trades = [t for t in trades if t['type'] == 'BUY']
    sell_trades = [t for t in trades if t['type'] == 'SELL']

    if sell_trades:
        profits = [t['profit_pct'] for t in sell_trades]
        win_rate = len([p for p in profits if p > 0]) / len(profits) * 100
        avg_profit = np.mean([p for p in profits if p > 0]) if any(p > 0 for p in profits) else 0
        avg_loss = np.mean([p for p in profits if p < 0]) if any(p < 0 for p in profits) else 0
    else:
        win_rate = 0
        avg_profit = 0
        avg_loss = 0

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'num_trades': len(buy_trades),
        'win_rate': win_rate,
        'avg_profit': avg_profit,
        'avg_loss': avg_loss
    }


def plot_comparison(results, symbol, save_path):
    """绘制策略对比图"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    # 1. 组合价值对比
    ax1 = axes[0, 0]
    for name, data in results.items():
        ax1.plot(data['portfolio']['value'], label=f"{name}", linewidth=1.5)
    ax1.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.5, label='Initial')
    ax1.set_title(f'{symbol} Portfolio Value Comparison', fontsize=12)
    ax1.set_ylabel('Value ($)')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # 2. 总收益对比柱状图
    ax2 = axes[0, 1]
    names = list(results.keys())
    returns = [results[n]['metrics']['total_return'] for n in names]
    colors = ['green' if r > 0 else 'red' for r in returns]
    bars = ax2.bar(names, returns, color=colors, alpha=0.7)
    ax2.set_title('Total Return Comparison (%)', fontsize=12)
    ax2.set_ylabel('Return (%)')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    for bar, ret in zip(bars, returns):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{ret:.1f}%', ha='center', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. 最大回撤对比
    ax3 = axes[1, 0]
    drawdowns = [results[n]['metrics']['max_drawdown'] for n in names]
    bars = ax3.bar(names, drawdowns, color='red', alpha=0.7)
    ax3.set_title('Max Drawdown Comparison (%)', fontsize=12)
    ax3.set_ylabel('Drawdown (%)')
    for bar, dd in zip(bars, drawdowns):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 2,
                f'{dd:.1f}%', ha='center', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. 夏普比率对比
    ax4 = axes[1, 1]
    sharpes = [results[n]['metrics']['sharpe'] for n in names]
    colors = ['green' if s > 0 else 'red' for s in sharpes]
    bars = ax4.bar(names, sharpes, color=colors, alpha=0.7)
    ax4.set_title('Sharpe Ratio Comparison', fontsize=12)
    ax4.set_ylabel('Sharpe Ratio')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    for bar, s in zip(bars, sharpes):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{s:.2f}', ha='center', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. 交易次数对比
    ax5 = axes[2, 0]
    num_trades = [results[n]['metrics']['num_trades'] for n in names]
    bars = ax5.bar(names, num_trades, color='blue', alpha=0.7)
    ax5.set_title('Number of Trades', fontsize=12)
    ax5.set_ylabel('Trades')
    for bar, n in zip(bars, num_trades):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{n}', ha='center', fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. 胜率对比
    ax6 = axes[2, 1]
    win_rates = [results[n]['metrics']['win_rate'] for n in names]
    bars = ax6.bar(names, win_rates, color='purple', alpha=0.7)
    ax6.set_title('Win Rate (%)', fontsize=12)
    ax6.set_ylabel('Win Rate (%)')
    ax6.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    for bar, wr in zip(bars, win_rates):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{wr:.1f}%', ha='center', fontsize=9)
    ax6.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {save_path}")


def main():
    """主函数"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("="*70)
    print("买入策略对比实验")
    print(f"股票: {TEST_SYMBOL}")
    print(f"期间: {TEST_START} ~ {TEST_END}")
    print(f"参数: Buy<{BUY_THRESHOLD}, AND>{AND_SELL_THRESHOLD}, OR>{OR_SELL_THRESHOLD}")
    print("="*70)

    # 准备数据
    print("\n加载数据...")
    df = prepare_data(TEST_SYMBOL, TEST_START, TEST_END)
    print(f"  数据范围: {df.index.min().date()} ~ {df.index.max().date()}")
    print(f"  数据点数: {len(df)}")

    # 运行各策略
    results = {}

    print("\n运行策略回测...")

    # 策略A: 基准
    print("\n  A. 基准策略 (一次性买入80%)...")
    final_a, trades_a, portfolio_a = strategy_a_baseline(
        df, BUY_THRESHOLD, AND_SELL_THRESHOLD, OR_SELL_THRESHOLD
    )
    metrics_a = calculate_metrics(portfolio_a, trades_a)
    results['A.基准'] = {'final': final_a, 'trades': trades_a, 'portfolio': portfolio_a, 'metrics': metrics_a}

    # 策略B: MA确认
    print("  B. MA确认策略 (sentiment<阈值 & price>MA10)...")
    final_b, trades_b, portfolio_b = strategy_b_ma_confirm(
        df, BUY_THRESHOLD, AND_SELL_THRESHOLD, OR_SELL_THRESHOLD
    )
    metrics_b = calculate_metrics(portfolio_b, trades_b)
    results['B.MA确认'] = {'final': final_b, 'trades': trades_b, 'portfolio': portfolio_b, 'metrics': metrics_b}

    # 策略C: 时间分批
    print("  C. 时间分批策略 (40% + 5天后40%)...")
    final_c, trades_c, portfolio_c = strategy_c_staged_time(
        df, BUY_THRESHOLD, AND_SELL_THRESHOLD, OR_SELL_THRESHOLD
    )
    metrics_c = calculate_metrics(portfolio_c, trades_c)
    results['C.时间分批'] = {'final': final_c, 'trades': trades_c, 'portfolio': portfolio_c, 'metrics': metrics_c}

    # 策略D: 阈值分批
    print("  D. 阈值分批策略 (4档各25%)...")
    final_d, trades_d, portfolio_d = strategy_d_staged_threshold(
        df, AND_SELL_THRESHOLD, OR_SELL_THRESHOLD
    )
    metrics_d = calculate_metrics(portfolio_d, trades_d)
    results['D.阈值分批'] = {'final': final_d, 'trades': trades_d, 'portfolio': portfolio_d, 'metrics': metrics_d}

    # 策略E: 反转确认
    print("  E. 反转确认策略 (等待sentiment回升)...")
    final_e, trades_e, portfolio_e = strategy_e_reversal_confirm(
        df, BUY_THRESHOLD, AND_SELL_THRESHOLD, OR_SELL_THRESHOLD
    )
    metrics_e = calculate_metrics(portfolio_e, trades_e)
    results['E.反转确认'] = {'final': final_e, 'trades': trades_e, 'portfolio': portfolio_e, 'metrics': metrics_e}

    # 打印结果对比
    print("\n" + "="*70)
    print("策略对比结果")
    print("="*70)
    print(f"\n{'策略':<12} {'总收益':>10} {'年化收益':>10} {'最大回撤':>10} {'夏普比率':>10} {'交易次数':>8} {'胜率':>8}")
    print("-"*70)

    for name, data in results.items():
        m = data['metrics']
        print(f"{name:<12} {m['total_return']:>+9.1f}% {m['annual_return']:>+9.1f}% "
              f"{m['max_drawdown']:>9.1f}% {m['sharpe']:>10.2f} {m['num_trades']:>8} {m['win_rate']:>7.1f}%")

    # 打印各策略交易详情
    print("\n" + "="*70)
    print("各策略交易详情")
    print("="*70)

    for name, data in results.items():
        print(f"\n{name}:")
        for t in data['trades'][:10]:  # 只显示前10笔
            if t['type'] == 'BUY':
                print(f"  {t['date'].strftime('%Y-%m-%d')} BUY  @ ${t['price']:.2f} | {t['reason']}")
            else:
                print(f"  {t['date'].strftime('%Y-%m-%d')} SELL @ ${t['price']:.2f} | {t['profit_pct']:+.1f}% | {t['reason']}")
        if len(data['trades']) > 10:
            print(f"  ... 共 {len(data['trades'])} 笔交易")

    # 保存结果
    # 汇总CSV
    summary_data = []
    for name, data in results.items():
        m = data['metrics']
        summary_data.append({
            'strategy': name,
            'final_value': data['final'],
            'total_return': m['total_return'],
            'annual_return': m['annual_return'],
            'max_drawdown': m['max_drawdown'],
            'sharpe': m['sharpe'],
            'num_trades': m['num_trades'],
            'win_rate': m['win_rate'],
            'avg_profit': m['avg_profit'],
            'avg_loss': m['avg_loss']
        })
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f'strategy_comparison_{TEST_SYMBOL}_{timestamp}.csv', index=False)

    # 所有交易记录
    all_trades = []
    for name, data in results.items():
        for t in data['trades']:
            t_copy = t.copy()
            t_copy['strategy'] = name
            t_copy['date'] = t_copy['date'].strftime('%Y-%m-%d')
            all_trades.append(t_copy)
    trades_df = pd.DataFrame(all_trades)
    trades_df.to_csv(f'trades_all_strategies_{TEST_SYMBOL}_{timestamp}.csv', index=False)

    # 绘制对比图
    plot_comparison(results, TEST_SYMBOL, f'strategy_comparison_{TEST_SYMBOL}_{timestamp}.png')

    print(f"\n输出文件:")
    print(f"  - strategy_comparison_{TEST_SYMBOL}_{timestamp}.csv")
    print(f"  - trades_all_strategies_{TEST_SYMBOL}_{timestamp}.csv")
    print(f"  - strategy_comparison_{TEST_SYMBOL}_{timestamp}.png")

    # 结论
    print("\n" + "="*70)
    print("分析结论")
    print("="*70)

    best_return = max(results.items(), key=lambda x: x[1]['metrics']['total_return'])
    best_sharpe = max(results.items(), key=lambda x: x[1]['metrics']['sharpe'])
    best_drawdown = max(results.items(), key=lambda x: x[1]['metrics']['max_drawdown'])  # 最小回撤

    print(f"\n  最高收益: {best_return[0]} ({best_return[1]['metrics']['total_return']:+.1f}%)")
    print(f"  最高夏普: {best_sharpe[0]} ({best_sharpe[1]['metrics']['sharpe']:.2f})")
    print(f"  最小回撤: {best_drawdown[0]} ({best_drawdown[1]['metrics']['max_drawdown']:.1f}%)")


if __name__ == "__main__":
    main()
