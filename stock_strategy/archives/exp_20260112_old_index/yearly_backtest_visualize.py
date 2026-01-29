"""
年度独立回测 - 保存交易记录并生成可视化图
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import psycopg2

# 配置中文字体 (必须在导入pyplot之前)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams['font.family'] = ['Heiti TC', 'STHeiti', 'PingFang HK', 'Songti SC']
plt.rcParams['axes.unicode_minus'] = False

from data_loader import DataLoader

db_config = {
    "host": "localhost",
    "port": 5432,
    "database": "crypto_fear_greed_2",
    "user": "sc2025",
    "password": ""
}

INITIAL_CAPITAL = 100000
COMMISSION_RATE = 0.001
SLIPPAGE_RATE = 0.001
POSITION_PCT = 0.8
MA_PERIOD = 50

# 最优配置
OPTIMAL_CONFIG = {
    'NVDA': {'buy_th': -10, 'sell_mode': 'and_ma', 'sell_th': 10},
    'TSLA': {'buy_th': -10, 'sell_mode': 'and_ma', 'sell_th': 10},
    'GOOGL': {'buy_th': -5, 'sell_mode': 'threshold', 'sell_th': 30},
    'META': {'buy_th': -10, 'sell_mode': 'and_ma', 'sell_th': 5},
    'MSFT': {'buy_th': -10, 'sell_mode': 'threshold', 'sell_th': 30},
    'AAPL': {'buy_th': -10, 'sell_mode': 'threshold', 'sell_th': 30},
    'AMZN': {'buy_th': -10, 'sell_mode': 'and_ma', 'sell_th': 5},
}

TEST_YEARS = [2021, 2022, 2023, 2024, 2025]


def load_sentiment_data(symbol):
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


def run_yearly_backtest(stock_data, sentiment_data, symbol, year):
    """单年度回测，返回详细交易信息"""
    config = OPTIMAL_CONFIG[symbol]
    buy_th = config['buy_th']
    sell_mode = config['sell_mode']
    sell_th = config['sell_th']

    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    lookback_start = pd.to_datetime(start_date, utc=True) - pd.Timedelta(days=MA_PERIOD * 2)
    end_dt = pd.to_datetime(end_date, utc=True)
    mask = (stock_data.index >= lookback_start) & (stock_data.index <= end_dt)
    price_data = stock_data[mask].copy()

    if len(price_data) < MA_PERIOD:
        return None

    price_data['MA'] = price_data['Close'].rolling(window=MA_PERIOD).mean()

    start_dt = pd.to_datetime(start_date, utc=True)
    test_mask = price_data.index >= start_dt
    price_data = price_data[test_mask].copy()

    if len(price_data) == 0:
        return None

    sent_data = sentiment_data.reindex(price_data.index)

    cash = INITIAL_CAPITAL
    shares = 0
    entry_price = 0
    entry_date = None
    entry_sentiment = None
    trades = []
    portfolio_values = []
    daily_returns = []
    prev_value = INITIAL_CAPITAL

    for i, (date, row) in enumerate(price_data.iterrows()):
        price = row['Close']
        ma = row['MA']
        sentiment = sent_data.loc[date, 'smoothed_index'] if date in sent_data.index else None

        current_total = cash + shares * price

        if pd.isna(sentiment) or pd.isna(ma):
            portfolio_values.append(current_total)
            daily_returns.append((current_total - prev_value) / prev_value if prev_value > 0 else 0)
            prev_value = current_total
            continue

        # 卖出逻辑
        if shares > 0:
            sell_signal = False
            exit_reason = None

            if sell_mode == 'threshold':
                if sentiment > sell_th:
                    sell_signal = True
                    exit_reason = f'sentiment>{sell_th}'
            elif sell_mode == 'and_ma':
                if sentiment > sell_th and price < ma:
                    sell_signal = True
                    exit_reason = f'>{sell_th}&<MA'

            if sell_signal:
                sell_price = price * (1 - SLIPPAGE_RATE) * (1 - COMMISSION_RATE)
                cash += shares * sell_price
                trades.append({
                    'symbol': symbol,
                    'year': year,
                    'type': 'SELL',
                    'date': date,
                    'price': price,
                    'net_price': sell_price,
                    'sentiment': sentiment,
                    'ma': ma,
                    'reason': exit_reason,
                    'shares': shares,
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'entry_sentiment': entry_sentiment,
                    'trade_return': (sell_price - entry_price) / entry_price
                })
                shares = 0
                entry_price = 0

        # 买入逻辑
        elif shares == 0 and sentiment < buy_th:
            position_value = cash * POSITION_PCT
            buy_price = price * (1 + SLIPPAGE_RATE) * (1 + COMMISSION_RATE)
            shares = int(position_value / buy_price)
            if shares > 0:
                cash -= shares * buy_price
                entry_price = buy_price
                entry_date = date
                entry_sentiment = sentiment
                trades.append({
                    'symbol': symbol,
                    'year': year,
                    'type': 'BUY',
                    'date': date,
                    'price': price,
                    'net_price': buy_price,
                    'sentiment': sentiment,
                    'ma': ma,
                    'reason': f'sentiment<{buy_th}',
                    'shares': shares,
                    'entry_date': None,
                    'entry_price': None,
                    'entry_sentiment': None,
                    'trade_return': None
                })

        current_total = cash + shares * price
        portfolio_values.append(current_total)
        daily_returns.append((current_total - prev_value) / prev_value if prev_value > 0 else 0)
        prev_value = current_total

    # 年末清仓
    if shares > 0:
        final_price = price_data.iloc[-1]['Close']
        final_sentiment = sent_data.iloc[-1]['smoothed_index'] if len(sent_data) > 0 else None
        final_ma = price_data.iloc[-1]['MA']
        sell_price = final_price * (1 - SLIPPAGE_RATE) * (1 - COMMISSION_RATE)
        cash += shares * sell_price
        trades.append({
            'symbol': symbol,
            'year': year,
            'type': 'SELL',
            'date': price_data.index[-1],
            'price': final_price,
            'net_price': sell_price,
            'sentiment': final_sentiment,
            'ma': final_ma,
            'reason': 'year_end',
            'shares': shares,
            'entry_date': entry_date,
            'entry_price': entry_price,
            'entry_sentiment': entry_sentiment,
            'trade_return': (sell_price - entry_price) / entry_price
        })

    final_value = cash
    total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL

    # 最大回撤
    if len(portfolio_values) > 0:
        portfolio_series = pd.Series(portfolio_values)
        rolling_max = portfolio_series.expanding().max()
        drawdowns = (portfolio_series - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
    else:
        max_drawdown = 0

    # 夏普率
    daily_returns_arr = np.array(daily_returns)
    if len(daily_returns_arr) > 0 and np.std(daily_returns_arr) > 0:
        sharpe = np.mean(daily_returns_arr) / np.std(daily_returns_arr) * np.sqrt(252)
    else:
        sharpe = 0

    return {
        'year': year,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'trades': trades,
        'final_value': final_value,
        'price_data': price_data
    }


def plot_trades(symbol, stock_data, all_trades, output_dir):
    """绘制单只股票的交易可视化图"""
    config = OPTIMAL_CONFIG[symbol]
    buy_th = config['buy_th']
    sell_th = config['sell_th']
    sell_mode = config['sell_mode']

    # 筛选该股票的交易
    trades = [t for t in all_trades if t['symbol'] == symbol]

    # 获取2021-2025的数据
    start_dt = pd.to_datetime('2021-01-01', utc=True)
    end_dt = pd.to_datetime('2025-12-31', utc=True)
    mask = (stock_data.index >= start_dt) & (stock_data.index <= end_dt)
    price_data = stock_data[mask].copy()
    price_data['MA'] = stock_data['Close'].rolling(window=MA_PERIOD).mean()

    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})

    # 上图：价格和交易信号
    ax1 = axes[0]
    ax1.plot(price_data.index, price_data['Close'], 'b-', linewidth=1.5, label='Price', alpha=0.8)
    ax1.plot(price_data.index, price_data['MA'], 'orange', linewidth=1, label='MA50', alpha=0.6)

    # 标记买入点
    buy_trades = [t for t in trades if t['type'] == 'BUY']
    for t in buy_trades:
        ax1.scatter(t['date'], t['price'], color='green', marker='^', s=150,
                   zorder=5, edgecolors='darkgreen', linewidths=1.5)
        ax1.annotate(f"Buy\n{t['sentiment']:.1f}",
                    (t['date'], t['price']),
                    textcoords="offset points", xytext=(0, 15),
                    ha='center', fontsize=8, color='green')

    # 标记卖出点
    sell_trades = [t for t in trades if t['type'] == 'SELL']
    for t in sell_trades:
        if t['reason'] == 'year_end':
            color = 'blue'
            label = 'YearEnd'
        elif '&<MA' in t['reason']:
            color = 'red'
            label = 'AND Stop'
        else:
            color = 'purple'
            label = 'Threshold'

        ax1.scatter(t['date'], t['price'], color=color, marker='v', s=150,
                   zorder=5, edgecolors='dark' + color if color != 'purple' else 'purple', linewidths=1.5)
        ret = t['trade_return']
        ax1.annotate(f"{label}\n{ret:+.1%}",
                    (t['date'], t['price']),
                    textcoords="offset points", xytext=(0, -25),
                    ha='center', fontsize=8, color=color)

    # 添加年份分隔线
    for year in [2022, 2023, 2024, 2025]:
        year_start = pd.to_datetime(f'{year}-01-01', utc=True)
        ax1.axvline(x=year_start, color='gray', linestyle='--', alpha=0.3)
        ax1.text(year_start, ax1.get_ylim()[1], str(year), fontsize=10, alpha=0.5)

    sell_desc = f">{sell_th}" if sell_mode == 'threshold' else f">{sell_th}&<MA"
    ax1.set_title(f'{symbol} - 年度独立回测交易记录 (2021-2025)\n买入: 情绪<{buy_th}, 卖出: {sell_desc}',
                 fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 下图：情绪指数
    ax2 = axes[1]
    sentiment_data = load_sentiment_data(symbol)
    sent_mask = (sentiment_data.index >= start_dt) & (sentiment_data.index <= end_dt)
    sent_plot = sentiment_data[sent_mask]

    ax2.plot(sent_plot.index, sent_plot['smoothed_index'], 'purple', linewidth=1, alpha=0.7)
    ax2.axhline(y=buy_th, color='green', linestyle='--', alpha=0.5, label=f'Buy threshold ({buy_th})')
    ax2.axhline(y=sell_th, color='red', linestyle='--', alpha=0.5, label=f'Sell threshold ({sell_th})')
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax2.fill_between(sent_plot.index, sent_plot['smoothed_index'], 0,
                     where=sent_plot['smoothed_index'] < buy_th, alpha=0.3, color='green', label='Fear zone')
    ax2.fill_between(sent_plot.index, sent_plot['smoothed_index'], 0,
                     where=sent_plot['smoothed_index'] > sell_th, alpha=0.3, color='red', label='Greed zone')

    ax2.set_ylabel('Sentiment Index', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 格式化x轴
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

    # 保存
    output_path = os.path.join(output_dir, f'{symbol}_yearly_trades.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存图表: {output_path}")


def main():
    print("=" * 80)
    print("年度独立回测 - 保存交易记录并生成可视化图")
    print("=" * 80)

    # 创建输出目录
    output_dir = 'yearly_backtest_results'
    os.makedirs(output_dir, exist_ok=True)

    loader = DataLoader(db_config)

    all_trades = []
    all_results = {}

    for symbol in OPTIMAL_CONFIG.keys():
        print(f"\n处理 {symbol}...")
        stock_data = loader.load_ohlcv(symbol, start_date="2014-01-01")
        sentiment_data = load_sentiment_data(symbol)

        if stock_data is None or len(stock_data) == 0:
            continue

        results = []
        for year in TEST_YEARS:
            result = run_yearly_backtest(stock_data, sentiment_data, symbol, year)
            if result:
                results.append(result)
                all_trades.extend(result['trades'])

        all_results[symbol] = results

        # 生成可视化图
        plot_trades(symbol, stock_data, all_trades, output_dir)

    loader.close()

    # 保存交易记录到CSV
    if all_trades:
        trades_df = pd.DataFrame(all_trades)

        # 格式化日期
        trades_df['date'] = pd.to_datetime(trades_df['date']).dt.strftime('%Y-%m-%d')
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date']).dt.strftime('%Y-%m-%d')

        # 选择需要的列并排序
        cols = ['symbol', 'year', 'type', 'date', 'price', 'sentiment', 'ma', 'reason',
                'shares', 'entry_date', 'entry_price', 'entry_sentiment', 'trade_return']
        trades_df = trades_df[cols]

        # 按股票和日期排序
        trades_df = trades_df.sort_values(['symbol', 'date'])

        csv_path = os.path.join(output_dir, 'all_trades.csv')
        trades_df.to_csv(csv_path, index=False)
        print(f"\n✓ 交易记录已保存: {csv_path}")

        # 打印交易统计
        print("\n" + "=" * 80)
        print("交易统计")
        print("=" * 80)

        for symbol in OPTIMAL_CONFIG.keys():
            symbol_trades = trades_df[trades_df['symbol'] == symbol]
            buy_count = len(symbol_trades[symbol_trades['type'] == 'BUY'])
            sell_trades = symbol_trades[symbol_trades['type'] == 'SELL']

            if len(sell_trades) > 0:
                wins = len(sell_trades[sell_trades['trade_return'] > 0])
                win_rate = wins / len(sell_trades) * 100
                avg_return = sell_trades['trade_return'].mean() * 100
                print(f"{symbol}: {buy_count}笔交易, 胜率 {win_rate:.0f}%, 平均收益 {avg_return:+.1f}%")

    # 生成年度汇总表
    print("\n" + "=" * 80)
    print("年度收益汇总")
    print("=" * 80)

    summary_data = []
    for symbol, results in all_results.items():
        row = {'symbol': symbol}
        for r in results:
            row[str(r['year'])] = r['total_return']
        row['累计'] = np.prod([1 + r['total_return'] for r in results]) - 1
        row['平均回撤'] = np.mean([r['max_drawdown'] for r in results])
        row['平均夏普'] = np.mean([r['sharpe'] for r in results])
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, 'yearly_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ 年度汇总已保存: {summary_path}")

    # 打印汇总表
    print(f"\n{'股票':<8}", end='')
    for year in TEST_YEARS:
        print(f"{year:>10}", end='')
    print(f"{'累计':>12}{'平均夏普':>10}")
    print("-" * 70)

    for row in summary_data:
        print(f"{row['symbol']:<8}", end='')
        for year in TEST_YEARS:
            print(f"{row[str(year)]:>10.1%}", end='')
        print(f"{row['累计']:>12.1%}{row['平均夏普']:>10.2f}")

    print("\n" + "=" * 80)
    print(f"完成! 所有文件保存在 {output_dir}/ 目录")
    print("=" * 80)


if __name__ == "__main__":
    main()
