"""
Walk-Forward 交易过程可视化
=====================================
可视化训练和测试期的交易过程，便于分析和观察

输出图表:
1. 价格图 + 买卖信号标记
2. 情绪指数 + 阈值线
3. 组合价值曲线
4. Walk-Forward窗口边界

使用方法:
    python visualize_wf_trades.py NVDA
    python visualize_wf_trades.py AAPL
"""

import sys
sys.path.insert(0, '/Users/sc2025/Desktop/Claude/Quantrade/src')

import pandas as pd
import numpy as np
import psycopg2
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
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

# 阈值分批买入参数
BUY_THRESHOLDS = [5, 0, -5, -10]
BATCH_PCT = 0.25

# 卖出参数搜索范围
AND_SELL_RANGE = [5, 10, 15, 20, 25]
OR_SELL_RANGE = [30, 40, 50, 55, 60]

# Walk-Forward 窗口
WINDOWS = [
    {"name": "W2020", "train": ("2016-01-01", "2019-12-31"), "test": ("2020-01-01", "2020-12-31")},
    {"name": "W2021", "train": ("2017-01-01", "2020-12-31"), "test": ("2021-01-01", "2021-12-31")},
    {"name": "W2022", "train": ("2018-01-01", "2021-12-31"), "test": ("2022-01-01", "2022-12-31")},
    {"name": "W2023", "train": ("2019-01-01", "2022-12-31"), "test": ("2023-01-01", "2023-12-31")},
    {"name": "W2024", "train": ("2020-01-01", "2023-12-31"), "test": ("2024-01-01", "2024-12-31")},
    {"name": "W2025", "train": ("2021-01-01", "2024-12-31"), "test": ("2025-01-01", "2025-12-31")},
]


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
# 阈值分批策略核心
# ============================================================

def run_threshold_staged(df, and_threshold, or_threshold,
                         cash, position, entry_price, bought_levels=None):
    """阈值分批买入策略"""
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

            if current_sentiment > or_threshold:
                sell_signal = True
                sell_reason = f"OR: {current_sentiment:.1f}>{or_threshold}"
            elif current_sentiment > and_threshold and current_price < current_ma50:
                sell_signal = True
                sell_reason = f"AND: {current_sentiment:.1f}>{and_threshold}"

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
                    'sentiment': current_sentiment,
                    'reason': sell_reason,
                    'profit_pct': profit_pct
                })

                position = 0
                entry_price = 0
                bought_levels = set()
                initial_capital_for_batch = cash

        # 买入逻辑
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
                        'sentiment': current_sentiment,
                        'reason': f"Batch{level_idx+1}(25%): sent {current_sentiment:.1f}<{threshold}",
                        'batch': level_idx + 1
                    })

        total_value = cash + position * current_price
        portfolio_values.append({
            'date': current_date,
            'value': total_value,
            'cash': cash,
            'position': position,
            'price': current_price,
            'sentiment': current_sentiment
        })

    final_value = cash + position * df['Close'].iloc[-1] if len(df) > 0 else cash
    portfolio_df = pd.DataFrame(portfolio_values).set_index('date') if portfolio_values else pd.DataFrame()

    return final_value, trades, portfolio_df, cash, position, entry_price, bought_levels


def grid_search(train_df):
    """网格搜索最优卖出参数"""
    best_return = -float('inf')
    best_params = None

    for and_t in AND_SELL_RANGE:
        for or_t in OR_SELL_RANGE:
            final_value, _, _, _, _, _, _ = run_threshold_staged(
                train_df, and_t, or_t, INITIAL_CAPITAL, 0, 0, None
            )
            ret = (final_value / INITIAL_CAPITAL - 1) * 100

            if ret > best_return:
                best_return = ret
                best_params = (and_t, or_t)

    return best_params, best_return


# ============================================================
# Walk-Forward 并记录详细数据
# ============================================================

def run_walk_forward_with_details(symbol):
    """运行Walk-Forward并记录详细数据用于可视化"""
    print(f"\n{'='*60}")
    print(f"  {symbol} - 阈值分批策略 Walk-Forward 可视化")
    print(f"{'='*60}")

    price_df = load_price(symbol)
    sentiment_df = load_sentiment_s3(symbol)

    cash = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    bought_levels = None

    all_trades = []
    all_portfolios = []
    window_details = []

    for window in WINDOWS:
        window_name = window['name']
        train_start, train_end = window['train']
        test_start, test_end = window['test']

        train_df = prepare_data(price_df, sentiment_df, train_start, train_end)
        test_df = prepare_data(price_df, sentiment_df, test_start, test_end)

        if len(train_df) < 100 or len(test_df) < 10:
            continue

        # 训练期网格搜索
        best_params, train_return = grid_search(train_df)
        and_t, or_t = best_params

        test_start_value = cash + position * test_df['Close'].iloc[0] if len(test_df) > 0 else cash

        # 测试期回测
        final_value, trades, portfolio_df, cash, position, entry_price, bought_levels = run_threshold_staged(
            test_df, and_t, or_t, cash, position, entry_price, bought_levels
        )

        test_return = (final_value / test_start_value - 1) * 100 if test_start_value > 0 else 0

        # 记录窗口详情
        window_details.append({
            'name': window_name,
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'and_threshold': and_t,
            'or_threshold': or_t,
            'train_return': train_return,
            'test_return': test_return
        })

        # 记录交易
        for t in trades:
            t['window'] = window_name
            all_trades.append(t)

        # 记录组合
        if not portfolio_df.empty:
            portfolio_df['window'] = window_name
            all_portfolios.append(portfolio_df)

        print(f"  {window_name}: AND>{and_t}, OR>{or_t} | Train: +{train_return:.1f}% | Test: {test_return:+.1f}%")

    # 合并组合数据
    if all_portfolios:
        full_portfolio = pd.concat(all_portfolios)
    else:
        full_portfolio = pd.DataFrame()

    total_return = (final_value / INITIAL_CAPITAL - 1) * 100
    print(f"\n  总收益: ${INITIAL_CAPITAL:,} → ${final_value:,.0f} ({total_return:+.1f}%)")

    return {
        'symbol': symbol,
        'price_df': price_df,
        'sentiment_df': sentiment_df,
        'trades': all_trades,
        'portfolio': full_portfolio,
        'window_details': window_details,
        'total_return': total_return,
        'final_value': final_value
    }


# ============================================================
# 可视化函数
# ============================================================

def create_visualization(data, output_file=None):
    """创建完整的可视化图表"""
    symbol = data['symbol']
    price_df = data['price_df']
    sentiment_df = data['sentiment_df']
    trades = data['trades']
    portfolio = data['portfolio']
    window_details = data['window_details']

    # 只显示测试期数据 (2020-2025)
    test_start = '2020-01-01'
    test_end = '2025-12-31'

    price_test = price_df[test_start:test_end]
    sentiment_test = sentiment_df[test_start:test_end]

    # 创建图表
    fig = plt.figure(figsize=(20, 16))

    # 使用GridSpec进行布局
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 2, 2, 1], hspace=0.15)

    # ========== 子图1: 价格 + 买卖信号 ==========
    ax1 = fig.add_subplot(gs[0])

    # 绘制价格线
    ax1.plot(price_test.index, price_test['Close'], 'b-', linewidth=1.5, alpha=0.8, label='Price')

    # 绘制MA50
    ma50 = price_test['Close'].rolling(50).mean()
    ax1.plot(price_test.index, ma50, 'gray', linewidth=1, alpha=0.6, linestyle='--', label='MA50')

    # 标记Walk-Forward窗口边界
    colors = ['#FFE4E1', '#E6E6FA', '#E0FFFF', '#F0FFF0', '#FFF8DC', '#FFE4B5']
    for i, w in enumerate(window_details):
        test_s = pd.Timestamp(w['test_start'], tz='UTC')
        test_e = pd.Timestamp(w['test_end'], tz='UTC')
        ax1.axvspan(test_s, test_e, alpha=0.15, color=colors[i % len(colors)],
                   label=f"{w['name']} Test" if i == 0 else "")

        # 在顶部标注窗口名称和参数
        mid_date = test_s + (test_e - test_s) / 2
        y_top = ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else price_test['Close'].max()
        ax1.text(mid_date, price_test['Close'].max() * 1.02,
                f"{w['name']}\nAND>{w['and_threshold']} OR>{w['or_threshold']}",
                ha='center', va='bottom', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 标记买入点 (不同批次不同颜色)
    batch_colors = ['#32CD32', '#228B22', '#006400', '#004d00']  # 浅绿到深绿
    batch_markers = ['^', '^', '^', '^']

    buy_trades = [t for t in trades if t['type'] == 'BUY']
    for t in buy_trades:
        batch = t.get('batch', 1) - 1
        color = batch_colors[min(batch, len(batch_colors)-1)]
        ax1.scatter(t['date'], t['price'],
                   color=color, marker='^', s=120, zorder=5,
                   edgecolors='black', linewidths=0.5)

    # 标记卖出点 (盈利红色，亏损橙色)
    sell_trades = [t for t in trades if t['type'] == 'SELL']
    for t in sell_trades:
        if t['profit_pct'] > 0:
            color = 'red'
            marker = 'v'
        else:
            color = 'orange'
            marker = 'v'
        ax1.scatter(t['date'], t['price'],
                   color=color, marker=marker, s=120, zorder=5,
                   edgecolors='black', linewidths=0.5)

    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title(f'{symbol} - Walk-Forward Trading Visualization (2020-2025)\n'
                  f'Total Return: {data["total_return"]:+.1f}% | Final Value: ${data["final_value"]:,.0f}',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 自定义图例
    legend_elements = [
        Line2D([0], [0], color='b', linewidth=2, label='Price'),
        Line2D([0], [0], color='gray', linewidth=1, linestyle='--', label='MA50'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#32CD32',
               markersize=10, markeredgecolor='black', label='Buy Batch 1 (sent<5)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#228B22',
               markersize=10, markeredgecolor='black', label='Buy Batch 2 (sent<0)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#006400',
               markersize=10, markeredgecolor='black', label='Buy Batch 3 (sent<-5)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#004d00',
               markersize=10, markeredgecolor='black', label='Buy Batch 4 (sent<-10)'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='red',
               markersize=10, markeredgecolor='black', label='Sell (Profit)'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='orange',
               markersize=10, markeredgecolor='black', label='Sell (Loss)'),
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=9, ncol=2)

    # ========== 子图2: 情绪指数 + 阈值线 ==========
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # 绘制情绪线
    ax2.plot(sentiment_test.index, sentiment_test['smoothed_index'],
             'purple', linewidth=1.5, alpha=0.8, label='Sentiment (S3)')

    # 绘制买入阈值线
    threshold_colors = ['#32CD32', '#228B22', '#006400', '#004d00']
    for i, thresh in enumerate(BUY_THRESHOLDS):
        ax2.axhline(y=thresh, color=threshold_colors[i], linestyle='--',
                   alpha=0.7, linewidth=1, label=f'Buy Level {i+1}: {thresh}')

    # 绘制零线
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)

    # 标记窗口边界
    for i, w in enumerate(window_details):
        test_s = pd.Timestamp(w['test_start'], tz='UTC')
        test_e = pd.Timestamp(w['test_end'], tz='UTC')
        ax2.axvspan(test_s, test_e, alpha=0.1, color=colors[i % len(colors)])

    # 标记买入点的情绪值
    for t in buy_trades:
        batch = t.get('batch', 1) - 1
        color = batch_colors[min(batch, len(batch_colors)-1)]
        ax2.scatter(t['date'], t['sentiment'],
                   color=color, marker='^', s=80, zorder=5,
                   edgecolors='black', linewidths=0.5)

    # 标记卖出点的情绪值
    for t in sell_trades:
        color = 'red' if t['profit_pct'] > 0 else 'orange'
        ax2.scatter(t['date'], t['sentiment'],
                   color=color, marker='v', s=80, zorder=5,
                   edgecolors='black', linewidths=0.5)

    ax2.set_ylabel('Sentiment Index', fontsize=12)
    ax2.set_ylim(-30, 70)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=9, ncol=2)

    # ========== 子图3: 组合价值 ==========
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    if not portfolio.empty:
        ax3.plot(portfolio.index, portfolio['value'], 'green', linewidth=2, label='Portfolio Value')
        ax3.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')

        # 标记窗口边界
        for i, w in enumerate(window_details):
            test_s = pd.Timestamp(w['test_start'], tz='UTC')
            test_e = pd.Timestamp(w['test_end'], tz='UTC')
            ax3.axvspan(test_s, test_e, alpha=0.1, color=colors[i % len(colors)])

        # 在卖出点标注收益
        for t in sell_trades:
            profit = t['profit_pct']
            color = 'green' if profit > 0 else 'red'
            ax3.annotate(f'{profit:+.1f}%',
                        xy=(t['date'], portfolio.loc[t['date'], 'value'] if t['date'] in portfolio.index else INITIAL_CAPITAL),
                        xytext=(0, 10), textcoords='offset points',
                        fontsize=8, color=color, ha='center',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    ax3.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left', fontsize=10)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # ========== 子图4: 交易统计表 ==========
    ax4 = fig.add_subplot(gs[3])
    ax4.axis('off')

    # 创建统计表格
    table_data = []
    for w in window_details:
        window_trades = [t for t in trades if t['window'] == w['name']]
        buys = len([t for t in window_trades if t['type'] == 'BUY'])
        sells = len([t for t in window_trades if t['type'] == 'SELL'])
        wins = len([t for t in window_trades if t['type'] == 'SELL' and t['profit_pct'] > 0])
        win_rate = (wins / sells * 100) if sells > 0 else 0

        table_data.append([
            w['name'],
            f"{w['train_start'][:4]}-{w['train_end'][:4]}",
            w['test_start'][:4],
            f">{w['and_threshold']}",
            f">{w['or_threshold']}",
            f"+{w['train_return']:.1f}%",
            f"{w['test_return']:+.1f}%",
            f"{buys}",
            f"{sells}",
            f"{win_rate:.0f}%"
        ])

    columns = ['Window', 'Train Period', 'Test', 'AND', 'OR',
               'Train Ret', 'Test Ret', 'Buys', 'Sells', 'Win Rate']

    table = ax4.table(cellText=table_data, colLabels=columns,
                      loc='center', cellLoc='center',
                      colWidths=[0.08, 0.12, 0.06, 0.06, 0.06, 0.1, 0.1, 0.06, 0.06, 0.08])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # 设置表头样式
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # 设置交替行颜色
    for i in range(1, len(table_data) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#D6DCE5')

    plt.tight_layout()

    # 保存图表
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'wf_visualization_{symbol}_{timestamp}.png'

    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n  图表已保存: {output_file}")

    plt.close()
    return output_file


def create_trade_detail_chart(data, output_file=None):
    """创建交易详情图表 - 每个窗口单独展示"""
    symbol = data['symbol']
    price_df = data['price_df']
    sentiment_df = data['sentiment_df']
    trades = data['trades']
    window_details = data['window_details']

    n_windows = len(window_details)
    fig, axes = plt.subplots(n_windows, 2, figsize=(20, 4 * n_windows))

    if n_windows == 1:
        axes = axes.reshape(1, -1)

    batch_colors = ['#32CD32', '#228B22', '#006400', '#004d00']

    for idx, w in enumerate(window_details):
        # 左图: 价格
        ax_price = axes[idx, 0]
        # 右图: 情绪
        ax_sent = axes[idx, 1]

        test_start = w['test_start']
        test_end = w['test_end']

        price_window = price_df[test_start:test_end]
        sentiment_window = sentiment_df[test_start:test_end]
        window_trades = [t for t in trades if t['window'] == w['name']]

        # ===== 左图: 价格 =====
        ax_price.plot(price_window.index, price_window['Close'], 'b-', linewidth=1.5)

        # MA50
        ma50 = price_window['Close'].rolling(50).mean()
        ax_price.plot(price_window.index, ma50, 'gray', linewidth=1, linestyle='--', alpha=0.6)

        # 买卖标记
        for t in window_trades:
            if t['type'] == 'BUY':
                batch = t.get('batch', 1) - 1
                color = batch_colors[min(batch, len(batch_colors)-1)]
                ax_price.scatter(t['date'], t['price'], color=color, marker='^', s=150,
                               zorder=5, edgecolors='black', linewidths=1)
                ax_price.annotate(f"B{batch+1}\n${t['price']:.1f}",
                                xy=(t['date'], t['price']),
                                xytext=(0, -25), textcoords='offset points',
                                fontsize=8, ha='center', color=color)
            else:
                color = 'red' if t['profit_pct'] > 0 else 'orange'
                ax_price.scatter(t['date'], t['price'], color=color, marker='v', s=150,
                               zorder=5, edgecolors='black', linewidths=1)
                ax_price.annotate(f"SELL\n{t['profit_pct']:+.1f}%",
                                xy=(t['date'], t['price']),
                                xytext=(0, 15), textcoords='offset points',
                                fontsize=8, ha='center', color=color)

        ax_price.set_title(f"{w['name']} Test Period ({test_start[:4]}) - Price", fontsize=11, fontweight='bold')
        ax_price.set_ylabel('Price ($)')
        ax_price.grid(True, alpha=0.3)
        ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

        # ===== 右图: 情绪 =====
        ax_sent.plot(sentiment_window.index, sentiment_window['smoothed_index'],
                    'purple', linewidth=1.5)

        # 阈值线
        for i, thresh in enumerate(BUY_THRESHOLDS):
            ax_sent.axhline(y=thresh, color=batch_colors[i], linestyle='--', alpha=0.7)

        # 卖出阈值
        ax_sent.axhline(y=w['and_threshold'], color='blue', linestyle=':', alpha=0.5,
                       label=f"AND>{w['and_threshold']}")
        ax_sent.axhline(y=w['or_threshold'], color='red', linestyle=':', alpha=0.5,
                       label=f"OR>{w['or_threshold']}")

        # 买卖标记
        for t in window_trades:
            if t['type'] == 'BUY':
                batch = t.get('batch', 1) - 1
                color = batch_colors[min(batch, len(batch_colors)-1)]
                ax_sent.scatter(t['date'], t['sentiment'], color=color, marker='^', s=100,
                              zorder=5, edgecolors='black', linewidths=0.5)
            else:
                color = 'red' if t['profit_pct'] > 0 else 'orange'
                ax_sent.scatter(t['date'], t['sentiment'], color=color, marker='v', s=100,
                              zorder=5, edgecolors='black', linewidths=0.5)

        ax_sent.set_title(f"{w['name']} Test Period ({test_start[:4]}) - Sentiment | "
                         f"Train: +{w['train_return']:.1f}% | Test: {w['test_return']:+.1f}%",
                         fontsize=11, fontweight='bold')
        ax_sent.set_ylabel('Sentiment')
        ax_sent.set_ylim(-30, 70)
        ax_sent.grid(True, alpha=0.3)
        ax_sent.legend(loc='upper right', fontsize=8)
        ax_sent.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

    plt.suptitle(f'{symbol} - Walk-Forward Trading Details by Window\n'
                 f'Buy Thresholds: {BUY_THRESHOLDS} | Total Return: {data["total_return"]:+.1f}%',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'wf_detail_{symbol}_{timestamp}.png'

    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  详情图已保存: {output_file}")

    plt.close()
    return output_file


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数"""
    # 解析命令行参数
    if len(sys.argv) > 1:
        symbol = sys.argv[1].upper()
    else:
        symbol = "NVDA"  # 默认

    print("="*60)
    print("Walk-Forward 交易过程可视化")
    print("="*60)
    print(f"股票: {symbol}")
    print(f"买入档位: {BUY_THRESHOLDS}")
    print(f"每档仓位: {BATCH_PCT*100:.0f}%")

    # 运行回测并收集数据
    data = run_walk_forward_with_details(symbol)

    # 创建可视化
    print("\n正在生成可视化图表...")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 1. 综合图
    overview_file = f'wf_visualization_{symbol}_{timestamp}.png'
    create_visualization(data, overview_file)

    # 2. 详情图
    detail_file = f'wf_detail_{symbol}_{timestamp}.png'
    create_trade_detail_chart(data, detail_file)

    print(f"\n完成！")


if __name__ == "__main__":
    main()
