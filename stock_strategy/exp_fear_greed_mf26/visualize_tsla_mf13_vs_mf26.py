"""
TSLA MF=13 vs MF=26 交易过程可视化
====================================
4行布局:
  Row 1: 价格 + MA50 + MF=13 买卖信号
  Row 2: MF=13 情绪指数 + 买入/卖出阈值线
  Row 3: 价格 + MA50 + MF=26 买卖信号
  Row 4: MF=26 情绪指数 + 买入/卖出阈值线

支持命令行参数:
  python visualize_tsla_mf13_vs_mf26.py [SYMBOL]
  默认: TSLA
"""

import sys
sys.path.insert(0, '/Users/sc2025/Desktop/Claude/Quantrade/src')

import pandas as pd
import numpy as np
import psycopg2
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from datetime import datetime
from data_loader import DataLoader

# ============================================================
# 配置
# ============================================================

DB_CONFIG = {
    "host": "localhost", "port": 5432,
    "database": "crypto_fear_greed_2",
    "user": "sc2025", "password": ""
}

INITIAL_CAPITAL = 100000
COMMISSION = 0.001
SLIPPAGE = 0.001
BUY_THRESHOLDS = [5, 0, -5, -10]
BATCH_PCT = 0.25
AND_SELL_RANGE_MF13 = [5, 10, 15, 20, 25]
AND_SELL_RANGE_MF26 = [15, 20, 25, 30]  # 优化版
OR_SELL_RANGE = [30, 40, 50, 55, 60]

WINDOWS = [
    {"name": "W2020", "train": ("2016-01-01", "2019-12-31"), "test": ("2020-01-01", "2020-12-31")},
    {"name": "W2021", "train": ("2017-01-01", "2020-12-31"), "test": ("2021-01-01", "2021-12-31")},
    {"name": "W2022", "train": ("2018-01-01", "2021-12-31"), "test": ("2022-01-01", "2022-12-31")},
    {"name": "W2023", "train": ("2019-01-01", "2022-12-31"), "test": ("2023-01-01", "2023-12-31")},
    {"name": "W2024", "train": ("2020-01-01", "2023-12-31"), "test": ("2024-01-01", "2024-12-31")},
    {"name": "W2025", "train": ("2021-01-01", "2024-12-31"), "test": ("2025-01-01", "2025-12-31")},
]

SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "TSLA"


# ============================================================
# 数据加载
# ============================================================

def load_sentiment(table, symbol):
    conn = psycopg2.connect(**DB_CONFIG)
    query = f"SELECT date, smoothed_index FROM {table} WHERE symbol = %s ORDER BY date"
    df = pd.read_sql(query, conn, params=(symbol,))
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.set_index('date')
    conn.close()
    return df


def load_price(symbol):
    loader = DataLoader(DB_CONFIG)
    price_df = loader.load_ohlcv(symbol, start_date="2011-01-01")
    loader.close()
    return price_df


def prepare_data(price_df, sentiment_df, start_date, end_date):
    df = price_df.copy()
    df['sentiment'] = sentiment_df['smoothed_index'].reindex(df.index)
    df['MA50'] = df['Close'].rolling(50).mean()
    df = df.dropna()
    return df[start_date:end_date]


# ============================================================
# 策略回测 (返回交易记录+每日状态)
# ============================================================

def run_threshold_staged(df, and_threshold, or_threshold,
                         cash, position, entry_price, bought_levels=None):
    if bought_levels is None:
        bought_levels = set()

    trades = []
    initial_capital_for_batch = cash + position * df['Close'].iloc[0] if len(df) > 0 and position > 0 else cash

    for i in range(len(df)):
        current_date = df.index[i]
        current_price = df['Close'].iloc[i]
        current_sentiment = df['sentiment'].iloc[i]
        current_ma50 = df['MA50'].iloc[i]

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
                cash += position * sell_price
                trades.append({
                    'type': 'SELL', 'date': current_date, 'price': current_price,
                    'shares': position, 'profit_pct': profit_pct, 'reason': sell_reason,
                    'and_threshold': and_threshold, 'or_threshold': or_threshold
                })
                position = 0
                entry_price = 0
                bought_levels = set()
                initial_capital_for_batch = cash

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
                        'type': 'BUY', 'date': current_date, 'price': current_price,
                        'shares': shares, 'batch': level_idx + 1,
                        'reason': f"L{level_idx+1}: sent {current_sentiment:.1f} < {threshold}",
                        'and_threshold': and_threshold, 'or_threshold': or_threshold
                    })

    final_value = cash + position * df['Close'].iloc[-1] if len(df) > 0 else cash
    return final_value, trades, cash, position, entry_price, bought_levels


def grid_search(train_df, and_range):
    best_return = -float('inf')
    best_params = None
    for and_t in and_range:
        for or_t in OR_SELL_RANGE:
            fv, _, _, _, _, _ = run_threshold_staged(
                train_df, and_t, or_t, INITIAL_CAPITAL, 0, 0, None
            )
            ret = (fv / INITIAL_CAPITAL - 1) * 100
            if ret > best_return:
                best_return = ret
                best_params = (and_t, or_t)
    return best_params, best_return


def run_walk_forward_full(symbol, sentiment_table, and_range):
    """运行完整Walk-Forward，返回所有交易记录和每个窗口的参数"""
    price_df = load_price(symbol)
    sentiment_df = load_sentiment(sentiment_table, symbol)

    cash = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    bought_levels = None

    all_trades = []
    window_params = []

    for window in WINDOWS:
        train_df = prepare_data(price_df, sentiment_df, window['train'][0], window['train'][1])
        test_df = prepare_data(price_df, sentiment_df, window['test'][0], window['test'][1])

        if len(train_df) < 100 or len(test_df) < 10:
            continue

        best_params, _ = grid_search(train_df, and_range)
        and_t, or_t = best_params

        test_start_value = cash + position * test_df['Close'].iloc[0] if len(test_df) > 0 else cash
        final_value, trades, cash, position, entry_price, bought_levels = run_threshold_staged(
            test_df, and_t, or_t, cash, position, entry_price, bought_levels
        )
        test_return = (final_value / test_start_value - 1) * 100 if test_start_value > 0 else 0

        window_params.append({
            'window': window['name'],
            'and': and_t, 'or': or_t,
            'return': test_return, 'end_value': final_value
        })
        all_trades.extend(trades)

    total_return = (final_value / INITIAL_CAPITAL - 1) * 100
    return all_trades, window_params, total_return, final_value


# ============================================================
# 可视化
# ============================================================

def plot_comparison(symbol):
    print(f"加载数据: {symbol}")
    price_df = load_price(symbol)
    sentiment_s3 = load_sentiment('fear_greed_index_s3', symbol)
    sentiment_mf26 = load_sentiment('fear_greed_index', symbol)

    # 准备测试期数据 (2020-2025)
    test_start = '2020-01-01'
    test_end = '2025-12-31'
    df_s3 = prepare_data(price_df, sentiment_s3, test_start, test_end)
    df_mf26 = prepare_data(price_df, sentiment_mf26, test_start, test_end)

    print("运行 MF=13 Walk-Forward...")
    trades_s3, params_s3, ret_s3, fv_s3 = run_walk_forward_full(
        symbol, 'fear_greed_index_s3', AND_SELL_RANGE_MF13)

    print("运行 MF=26 Walk-Forward...")
    trades_mf26, params_mf26, ret_mf26, fv_mf26 = run_walk_forward_full(
        symbol, 'fear_greed_index', AND_SELL_RANGE_MF26)

    # 打印窗口参数
    print(f"\nMF=13 窗口参数:")
    for p in params_s3:
        print(f"  {p['window']}: AND>{p['and']}, OR>{p['or']} → {p['return']:+.1f}%")
    print(f"  总收益: +{ret_s3:.1f}% (${fv_s3:,.0f})")

    print(f"\nMF=26 窗口参数:")
    for p in params_mf26:
        print(f"  {p['window']}: AND>{p['and']}, OR>{p['or']} → {p['return']:+.1f}%")
    print(f"  总收益: +{ret_mf26:.1f}% (${fv_mf26:,.0f})")

    # ========== 绘图 ==========
    fig, axes = plt.subplots(4, 1, figsize=(28, 20), sharex=True,
                             gridspec_kw={'height_ratios': [3, 2, 3, 2]})
    fig.suptitle(f'{symbol} Walk-Forward 交易对比: MF=13 vs MF=26 (2020-2025)',
                 fontsize=18, fontweight='bold', y=0.98)

    # 窗口分界线和背景
    window_boundaries = [
        pd.Timestamp('2020-01-01', tz='UTC'),
        pd.Timestamp('2021-01-01', tz='UTC'),
        pd.Timestamp('2022-01-01', tz='UTC'),
        pd.Timestamp('2023-01-01', tz='UTC'),
        pd.Timestamp('2024-01-01', tz='UTC'),
        pd.Timestamp('2025-01-01', tz='UTC'),
        pd.Timestamp('2025-12-31', tz='UTC'),
    ]
    window_colors = ['#f0f8ff', '#fff8f0', '#f8f0f0', '#f0fff0', '#f8f0ff', '#fffff0']
    window_labels = ['W2020', 'W2021', 'W2022', 'W2023', 'W2024', 'W2025']

    for ax in axes:
        for i in range(len(window_boundaries) - 1):
            ax.axvspan(window_boundaries[i], window_boundaries[i+1],
                      alpha=0.3, color=window_colors[i], zorder=0)
            if ax == axes[0]:
                mid = window_boundaries[i] + (window_boundaries[i+1] - window_boundaries[i]) / 2
                ax.text(mid, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
                       window_labels[i], ha='center', va='top',
                       fontsize=10, color='gray', alpha=0.7)

    # ---------- Row 1: 价格 + MF=13 交易 ----------
    ax1 = axes[0]
    ax1.plot(df_s3.index, df_s3['Close'], color='#333333', linewidth=1.2, alpha=0.9, label='Price')
    ax1.plot(df_s3.index, df_s3['MA50'], color='orange', linewidth=1, alpha=0.6, linestyle='--', label='MA50')

    buys_s3 = [t for t in trades_s3 if t['type'] == 'BUY']
    sells_s3 = [t for t in trades_s3 if t['type'] == 'SELL']

    for t in buys_s3:
        ax1.scatter(t['date'], t['price'], color='limegreen', marker='^', s=120,
                   edgecolors='darkgreen', linewidths=1.2, zorder=5)
    for t in sells_s3:
        color = 'red' if t['profit_pct'] > 0 else 'orange'
        ax1.scatter(t['date'], t['price'], color=color, marker='v', s=120,
                   edgecolors='darkred' if t['profit_pct'] > 0 else 'darkorange',
                   linewidths=1.2, zorder=5)

    # 窗口参数标注
    for p in params_s3:
        w_start = [w for w in WINDOWS if w['name'] == p['window']][0]['test'][0]
        ax1.text(pd.Timestamp(w_start, tz='UTC') + pd.Timedelta(days=5),
                ax1.get_ylim()[0] * 0.95 if ax1.get_ylim()[0] > 0 else 0,
                f"AND>{p['and']} OR>{p['or']}\n{p['return']:+.1f}%",
                fontsize=7.5, color='#2255aa', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    ax1.set_ylabel('Price ($)', fontsize=11)
    ax1.set_title(f'MF=13 (fear_greed_index_s3) — 总收益: +{ret_s3:.1f}% (${fv_s3:,.0f})',
                  fontsize=13, color='#2255aa', loc='left')

    legend_elements_1 = [
        Line2D([0], [0], color='#333333', linewidth=1.2, label='Price'),
        Line2D([0], [0], color='orange', linewidth=1, linestyle='--', alpha=0.6, label='MA50'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='limegreen',
               markersize=10, markeredgecolor='darkgreen', markeredgewidth=1.2,
               label=f'Buy ({len(buys_s3)})', linestyle='None'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='red',
               markersize=10, markeredgecolor='darkred', markeredgewidth=1.2,
               label=f'Sell ({len(sells_s3)})', linestyle='None'),
    ]
    ax1.legend(handles=legend_elements_1, loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.2)

    # ---------- Row 2: MF=13 情绪指数 ----------
    ax2 = axes[1]
    ax2.plot(df_s3.index, df_s3['sentiment'], color='#2255aa', linewidth=1, alpha=0.8)
    ax2.axhline(y=5, color='green', linewidth=0.8, linestyle='--', alpha=0.5, label='Buy <5')
    ax2.axhline(y=0, color='green', linewidth=0.8, linestyle='-.', alpha=0.4, label='Buy <0')
    ax2.axhline(y=-5, color='green', linewidth=0.8, linestyle=':', alpha=0.4, label='Buy <-5')
    ax2.axhline(y=-10, color='green', linewidth=0.8, linestyle=':', alpha=0.3, label='Buy <-10')
    ax2.fill_between(df_s3.index, df_s3['sentiment'], 5,
                     where=df_s3['sentiment'] < 5, alpha=0.15, color='green')

    # 标注卖出阈值 (每个窗口不同)
    for p in params_s3:
        w = [w for w in WINDOWS if w['name'] == p['window']][0]
        ws = pd.Timestamp(w['test'][0], tz='UTC')
        we = pd.Timestamp(w['test'][1], tz='UTC')
        ax2.hlines(y=p['or'], xmin=ws, xmax=we, color='red', linewidth=1.2,
                  linestyle='-', alpha=0.6)
        ax2.hlines(y=p['and'], xmin=ws, xmax=we, color='red', linewidth=0.8,
                  linestyle='--', alpha=0.4)
        ax2.text(ws + pd.Timedelta(days=3), p['or'] + 2,
                f"OR>{p['or']}", fontsize=7, color='red', alpha=0.7)
        ax2.text(ws + pd.Timedelta(days=3), p['and'] - 4,
                f"AND>{p['and']}", fontsize=7, color='red', alpha=0.5)

    for t in buys_s3:
        ax2.axvline(x=t['date'], color='green', linewidth=0.5, alpha=0.3)
    for t in sells_s3:
        ax2.axvline(x=t['date'], color='red', linewidth=0.5, alpha=0.3)

    ax2.set_ylabel('Sentiment (MF=13)', fontsize=11)
    ax2.set_ylim(-50, 80)
    ax2.grid(True, alpha=0.2)

    # ---------- Row 3: 价格 + MF=26 交易 ----------
    ax3 = axes[2]
    ax3.plot(df_mf26.index, df_mf26['Close'], color='#333333', linewidth=1.2, alpha=0.9, label='Price')
    ax3.plot(df_mf26.index, df_mf26['MA50'], color='orange', linewidth=1, alpha=0.6, linestyle='--', label='MA50')

    buys_mf26 = [t for t in trades_mf26 if t['type'] == 'BUY']
    sells_mf26 = [t for t in trades_mf26 if t['type'] == 'SELL']

    for t in buys_mf26:
        ax3.scatter(t['date'], t['price'], color='cyan', marker='^', s=120,
                   edgecolors='darkcyan', linewidths=1.2, zorder=5)
    for t in sells_mf26:
        color = 'red' if t.get('profit_pct', 0) > 0 else 'orange'
        ax3.scatter(t['date'], t['price'], color=color, marker='v', s=120,
                   edgecolors='darkred' if t.get('profit_pct', 0) > 0 else 'darkorange',
                   linewidths=1.2, zorder=5)

    for p in params_mf26:
        w_start = [w for w in WINDOWS if w['name'] == p['window']][0]['test'][0]
        ax3.text(pd.Timestamp(w_start, tz='UTC') + pd.Timedelta(days=5),
                ax3.get_ylim()[0] * 0.95 if ax3.get_ylim()[0] > 0 else 0,
                f"AND>{p['and']} OR>{p['or']}\n{p['return']:+.1f}%",
                fontsize=7.5, color='#aa5522', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    ax3.set_ylabel('Price ($)', fontsize=11)
    ax3.set_title(f'MF=26 (fear_greed_index, AND>=15) — 总收益: +{ret_mf26:.1f}% (${fv_mf26:,.0f})',
                  fontsize=13, color='#aa5522', loc='left')

    legend_elements_3 = [
        Line2D([0], [0], color='#333333', linewidth=1.2, label='Price'),
        Line2D([0], [0], color='orange', linewidth=1, linestyle='--', alpha=0.6, label='MA50'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='cyan',
               markersize=10, markeredgecolor='darkcyan', markeredgewidth=1.2,
               label=f'Buy ({len(buys_mf26)})', linestyle='None'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='red',
               markersize=10, markeredgecolor='darkred', markeredgewidth=1.2,
               label=f'Sell ({len(sells_mf26)})', linestyle='None'),
    ]
    ax3.legend(handles=legend_elements_3, loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.2)

    # ---------- Row 4: MF=26 情绪指数 ----------
    ax4 = axes[3]
    ax4.plot(df_mf26.index, df_mf26['sentiment'], color='#aa5522', linewidth=1, alpha=0.8)
    ax4.axhline(y=5, color='green', linewidth=0.8, linestyle='--', alpha=0.5, label='Buy <5')
    ax4.axhline(y=0, color='green', linewidth=0.8, linestyle='-.', alpha=0.4, label='Buy <0')
    ax4.axhline(y=-5, color='green', linewidth=0.8, linestyle=':', alpha=0.4, label='Buy <-5')
    ax4.axhline(y=-10, color='green', linewidth=0.8, linestyle=':', alpha=0.3, label='Buy <-10')
    ax4.fill_between(df_mf26.index, df_mf26['sentiment'], 5,
                     where=df_mf26['sentiment'] < 5, alpha=0.15, color='green')

    for p in params_mf26:
        w = [w for w in WINDOWS if w['name'] == p['window']][0]
        ws = pd.Timestamp(w['test'][0], tz='UTC')
        we = pd.Timestamp(w['test'][1], tz='UTC')
        ax4.hlines(y=p['or'], xmin=ws, xmax=we, color='red', linewidth=1.2,
                  linestyle='-', alpha=0.6)
        ax4.hlines(y=p['and'], xmin=ws, xmax=we, color='red', linewidth=0.8,
                  linestyle='--', alpha=0.4)
        ax4.text(ws + pd.Timedelta(days=3), p['or'] + 2,
                f"OR>{p['or']}", fontsize=7, color='red', alpha=0.7)
        ax4.text(ws + pd.Timedelta(days=3), p['and'] - 4,
                f"AND>{p['and']}", fontsize=7, color='red', alpha=0.5)

    for t in buys_mf26:
        ax4.axvline(x=t['date'], color='green', linewidth=0.5, alpha=0.3)
    for t in sells_mf26:
        ax4.axvline(x=t['date'], color='red', linewidth=0.5, alpha=0.3)

    ax4.set_ylabel('Sentiment (MF=26)', fontsize=11)
    ax4.set_ylim(-50, 80)
    ax4.grid(True, alpha=0.2)

    # X 轴格式
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 窗口标签 (在第一个子图顶部)
    for i in range(len(window_boundaries) - 1):
        mid = window_boundaries[i] + (window_boundaries[i+1] - window_boundaries[i]) / 2
        axes[0].annotate(window_labels[i], xy=(mid, 1.02), xycoords=('data', 'axes fraction'),
                        ha='center', fontsize=10, color='gray', fontweight='bold')

    # 关键事件标注: 2021-12-31 分歧点
    key_date = pd.Timestamp('2021-12-31', tz='UTC')
    for ax in axes:
        ax.axvline(x=key_date, color='purple', linewidth=1.5, linestyle='-', alpha=0.4)

    axes[1].annotate('MF13=26.9 > AND25 → SELL',
                    xy=(key_date, 26.9), xytext=(key_date + pd.Timedelta(days=30), 50),
                    fontsize=8, color='purple', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='purple', lw=1.2))

    axes[3].annotate('MF26=24.4 < AND25 → HOLD',
                    xy=(key_date, 24.4), xytext=(key_date + pd.Timedelta(days=30), 50),
                    fontsize=8, color='purple', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='purple', lw=1.2))

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_file = f'{SYMBOL}_mf13_vs_mf26_trades.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存: {output_file}")

    # 打印交易对比
    print(f"\n{'='*60}")
    print(f"{SYMBOL} 交易统计")
    print(f"{'='*60}")
    print(f"{'':>20} {'MF=13':>12} {'MF=26':>12}")
    print(f"{'总收益':>20} {ret_s3:>+11.1f}% {ret_mf26:>+11.1f}%")
    print(f"{'最终资产':>20} ${fv_s3:>10,.0f} ${fv_mf26:>10,.0f}")
    print(f"{'买入次数':>20} {len(buys_s3):>12} {len(buys_mf26):>12}")
    print(f"{'卖出次数':>20} {len(sells_s3):>12} {len(sells_mf26):>12}")
    win_s3 = len([t for t in sells_s3 if t.get('profit_pct', 0) > 0])
    win_mf26 = len([t for t in sells_mf26 if t.get('profit_pct', 0) > 0])
    print(f"{'盈利卖出':>20} {win_s3:>12} {win_mf26:>12}")


if __name__ == "__main__":
    plot_comparison(SYMBOL)
