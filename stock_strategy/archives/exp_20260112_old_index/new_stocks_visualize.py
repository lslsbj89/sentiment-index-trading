"""
新股票回测可视化
UBER, HOOD, COIN, BABA
使用最优配置生成交易记录和可视化图表
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import psycopg2

# 配置中文字体
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

# 新股票最优配置
OPTIMAL_CONFIG = {
    'HOOD': {'buy_th': -5, 'sell_mode': 'and_ma', 'sell_th': 5, 'position_mode': 'dynamic'},
    'COIN': {'buy_th': -10, 'sell_mode': 'threshold', 'sell_th': 30, 'position_mode': 'fixed'},
    'UBER': {'buy_th': -5, 'sell_mode': 'and_ma', 'sell_th': 5, 'position_mode': 'dynamic'},
    'BABA': {'buy_th': -10, 'sell_mode': 'threshold', 'sell_th': 30, 'position_mode': 'fixed'},
}


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
    if len(df) == 0:
        return None
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.set_index('date')
    return df


def run_backtest(stock_data, sentiment_data, symbol):
    """运行回测"""
    config = OPTIMAL_CONFIG[symbol]
    buy_th = config['buy_th']
    sell_mode = config['sell_mode']
    sell_th = config['sell_th']
    use_dynamic = config['position_mode'] == 'dynamic'

    start_date = "2021-01-01"
    end_date = "2025-12-31"

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

    if len(price_data) == 0 or sentiment_data is None:
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
    fixed_position_value = INITIAL_CAPITAL * POSITION_PCT

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
                proceeds = shares * sell_price
                cash += proceeds
                trade_return = (sell_price - entry_price) / entry_price
                trades.append({
                    'symbol': symbol,
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
                    'trade_return': trade_return,
                    'portfolio_value': cash
                })
                shares = 0
                entry_price = 0

        # 买入逻辑
        elif shares == 0 and sentiment < buy_th:
            if use_dynamic:
                position_value = cash * POSITION_PCT
            else:
                position_value = min(fixed_position_value, cash)

            buy_price = price * (1 + SLIPPAGE_RATE) * (1 + COMMISSION_RATE)
            shares = int(position_value / buy_price)

            if shares > 0:
                cost = shares * buy_price
                if cost <= cash:
                    cash -= cost
                    entry_price = buy_price
                    entry_date = date
                    entry_sentiment = sentiment
                    trades.append({
                        'symbol': symbol,
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
                        'trade_return': None,
                        'portfolio_value': cash + shares * price
                    })
                else:
                    shares = 0

        current_total = cash + shares * price
        portfolio_values.append(current_total)
        daily_returns.append((current_total - prev_value) / prev_value if prev_value > 0 else 0)
        prev_value = current_total

    # 最终清仓
    if shares > 0:
        final_price = price_data.iloc[-1]['Close']
        final_sentiment = sent_data.iloc[-1]['smoothed_index'] if len(sent_data) > 0 else None
        final_ma = price_data.iloc[-1]['MA']
        sell_price = final_price * (1 - SLIPPAGE_RATE) * (1 - COMMISSION_RATE)
        proceeds = shares * sell_price
        cash += proceeds
        trade_return = (sell_price - entry_price) / entry_price
        trades.append({
            'symbol': symbol,
            'type': 'SELL',
            'date': price_data.index[-1],
            'price': final_price,
            'net_price': sell_price,
            'sentiment': final_sentiment,
            'ma': final_ma,
            'reason': 'end',
            'shares': shares,
            'entry_date': entry_date,
            'entry_price': entry_price,
            'entry_sentiment': entry_sentiment,
            'trade_return': trade_return,
            'portfolio_value': cash
        })

    final_value = cash
    total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL

    # 最大回撤
    if len(portfolio_values) > 0:
        portfolio_series = pd.Series(portfolio_values, index=price_data.index)
        rolling_max = portfolio_series.expanding().max()
        drawdowns = (portfolio_series - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
    else:
        max_drawdown = 0
        portfolio_series = pd.Series()

    # 年化收益
    years = 5
    annualized = (1 + total_return) ** (1/years) - 1 if total_return > -1 else -1

    # 夏普率
    daily_returns_arr = np.array(daily_returns)
    if len(daily_returns_arr) > 0 and np.std(daily_returns_arr) > 0:
        sharpe = np.mean(daily_returns_arr) / np.std(daily_returns_arr) * np.sqrt(252)
    else:
        sharpe = 0

    # 胜率
    sell_trades = [t for t in trades if t['type'] == 'SELL']
    if len(sell_trades) > 0:
        wins = sum(1 for t in sell_trades if t['trade_return'] and t['trade_return'] > 0)
        win_rate = wins / len(sell_trades)
    else:
        win_rate = 0

    return {
        'symbol': symbol,
        'total_return': total_return,
        'annualized': annualized,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'trades': trades,
        'final_value': final_value,
        'win_rate': win_rate,
        'portfolio_series': portfolio_series,
        'price_data': price_data
    }


def plot_trades(symbol, result, output_dir):
    """绘制交易可视化图"""
    config = OPTIMAL_CONFIG[symbol]
    buy_th = config['buy_th']
    sell_th = config['sell_th']
    sell_mode = config['sell_mode']
    position_mode = config['position_mode']

    trades = result['trades']
    price_data = result['price_data']
    portfolio_series = result['portfolio_series']

    # 创建图表
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), gridspec_kw={'height_ratios': [3, 2, 1]})

    # 上图：价格和交易信号
    ax1 = axes[0]
    ax1.plot(price_data.index, price_data['Close'], 'b-', linewidth=1.5, label='Price', alpha=0.8)
    ax1.plot(price_data.index, price_data['MA'], 'orange', linewidth=1, label='MA50', alpha=0.6)

    # 标记买入点
    buy_trades = [t for t in trades if t['type'] == 'BUY']
    for t in buy_trades:
        ax1.scatter(t['date'], t['price'], color='green', marker='^', s=150,
                   zorder=5, edgecolors='darkgreen', linewidths=1.5)
        ax1.annotate(f"买入\n{t['sentiment']:.1f}",
                    (t['date'], t['price']),
                    textcoords="offset points", xytext=(0, 15),
                    ha='center', fontsize=8, color='green')

    # 标记卖出点
    sell_trades = [t for t in trades if t['type'] == 'SELL']
    for t in sell_trades:
        if t['reason'] == 'end':
            color = 'blue'
            label = '清仓'
        elif '&<MA' in t['reason']:
            color = 'red'
            label = 'AND止损'
        else:
            color = 'purple'
            label = '阈值卖出'

        ax1.scatter(t['date'], t['price'], color=color, marker='v', s=150,
                   zorder=5, edgecolors='dark' + color if color != 'purple' else 'purple', linewidths=1.5)
        ret = t['trade_return'] if t['trade_return'] else 0
        ax1.annotate(f"{label}\n{ret:+.1%}",
                    (t['date'], t['price']),
                    textcoords="offset points", xytext=(0, -25),
                    ha='center', fontsize=8, color=color)

    # 添加年份分隔线
    for year in [2022, 2023, 2024, 2025]:
        year_start = pd.to_datetime(f'{year}-01-01', utc=True)
        if year_start <= price_data.index.max():
            ax1.axvline(x=year_start, color='gray', linestyle='--', alpha=0.3)
            ax1.text(year_start, ax1.get_ylim()[1] * 0.95, str(year), fontsize=10, alpha=0.5)

    sell_desc = f">{sell_th}" if sell_mode == 'threshold' else f">{sell_th}&<MA"
    pos_desc = "动态复利" if position_mode == 'dynamic' else "固定仓位"
    ax1.set_title(f'{symbol} - 连续5年回测 (2021-2025) - {pos_desc}\n'
                 f'买入: 情绪<{buy_th}, 卖出: {sell_desc} | '
                 f'收益: {result["total_return"]:.1%} | 夏普: {result["sharpe"]:.2f}',
                 fontsize=14, fontweight='bold')
    ax1.set_ylabel('价格 ($)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 中图：组合价值
    ax2 = axes[1]
    ax2.plot(portfolio_series.index, portfolio_series.values, 'green', linewidth=1.5, label='组合价值')
    ax2.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.5, label='初始资金')
    ax2.fill_between(portfolio_series.index, INITIAL_CAPITAL, portfolio_series.values,
                     where=portfolio_series.values >= INITIAL_CAPITAL, alpha=0.3, color='green')
    ax2.fill_between(portfolio_series.index, INITIAL_CAPITAL, portfolio_series.values,
                     where=portfolio_series.values < INITIAL_CAPITAL, alpha=0.3, color='red')

    ax2.set_ylabel('组合价值 ($)', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f'组合价值变化 | 初始: ${INITIAL_CAPITAL:,.0f} → 最终: ${result["final_value"]:,.0f} | '
                 f'最大回撤: {result["max_drawdown"]:.1%}', fontsize=12)

    # 下图：情绪指数
    ax3 = axes[2]
    sentiment_data = load_sentiment_data(symbol)
    if sentiment_data is not None:
        start_dt = pd.to_datetime('2021-01-01', utc=True)
        end_dt = pd.to_datetime('2025-12-31', utc=True)
        sent_mask = (sentiment_data.index >= start_dt) & (sentiment_data.index <= end_dt)
        sent_plot = sentiment_data[sent_mask]

        ax3.plot(sent_plot.index, sent_plot['smoothed_index'], 'purple', linewidth=1, alpha=0.7)
        ax3.axhline(y=buy_th, color='green', linestyle='--', alpha=0.5, label=f'买入阈值 ({buy_th})')
        ax3.axhline(y=sell_th, color='red', linestyle='--', alpha=0.5, label=f'卖出阈值 ({sell_th})')
        ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax3.fill_between(sent_plot.index, sent_plot['smoothed_index'], 0,
                         where=sent_plot['smoothed_index'] < buy_th, alpha=0.3, color='green', label='恐惧区')
        ax3.fill_between(sent_plot.index, sent_plot['smoothed_index'], 0,
                         where=sent_plot['smoothed_index'] > sell_th, alpha=0.3, color='red', label='贪婪区')

    ax3.set_ylabel('情绪指数', fontsize=12)
    ax3.set_xlabel('日期', fontsize=12)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 格式化x轴
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

    # 保存
    output_path = os.path.join(output_dir, f'{symbol}_trades.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存图表: {output_path}")


def main():
    print("=" * 80)
    print("新股票回测可视化 - UBER, HOOD, COIN, BABA")
    print("=" * 80)

    # 创建输出目录
    output_dir = 'new_stocks_results'
    os.makedirs(output_dir, exist_ok=True)

    loader = DataLoader(db_config)

    all_trades = []
    all_results = []

    for symbol in OPTIMAL_CONFIG.keys():
        print(f"\n处理 {symbol}...")
        stock_data = loader.load_ohlcv(symbol, start_date="2014-01-01")
        sentiment_data = load_sentiment_data(symbol)

        if stock_data is None or len(stock_data) == 0:
            print(f"  ⚠ {symbol} 无价格数据")
            continue

        if sentiment_data is None or len(sentiment_data) == 0:
            print(f"  ⚠ {symbol} 无情绪数据")
            continue

        result = run_backtest(stock_data, sentiment_data, symbol)
        if result:
            all_results.append(result)
            all_trades.extend(result['trades'])

            # 生成可视化图
            plot_trades(symbol, result, output_dir)

    loader.close()

    # 保存交易记录到CSV
    if all_trades:
        trades_df = pd.DataFrame(all_trades)

        # 格式化日期
        trades_df['date'] = pd.to_datetime(trades_df['date']).dt.strftime('%Y-%m-%d')
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date']).dt.strftime('%Y-%m-%d')

        # 选择需要的列并排序
        cols = ['symbol', 'type', 'date', 'price', 'sentiment', 'ma', 'reason',
                'shares', 'entry_date', 'entry_price', 'entry_sentiment', 'trade_return', 'portfolio_value']
        trades_df = trades_df[cols]

        # 按股票和日期排序
        trades_df = trades_df.sort_values(['symbol', 'date'])

        csv_path = os.path.join(output_dir, 'all_trades.csv')
        trades_df.to_csv(csv_path, index=False)
        print(f"\n✓ 交易记录已保存: {csv_path}")

    # 打印汇总结果
    print("\n" + "=" * 100)
    print("新股票回测汇总")
    print("=" * 100)

    print(f"\n{'股票':<8} {'配置':<25} {'5年收益':>12} {'年化':>10} {'回撤':>10} {'夏普':>8} {'交易':>6} {'胜率':>8}")
    print("-" * 95)

    for r in sorted(all_results, key=lambda x: x['total_return'], reverse=True):
        config = OPTIMAL_CONFIG[r['symbol']]
        sell_short = f">{config['sell_th']}" if config['sell_mode'] == 'threshold' else f">{config['sell_th']}&MA"
        config_str = f"<{config['buy_th']}, {sell_short}"
        sell_trades = [t for t in r['trades'] if t['type'] == 'SELL']

        print(f"{r['symbol']:<8} {config_str:<25} {r['total_return']:>12.2%} "
              f"{r['annualized']:>10.2%} {r['max_drawdown']:>10.2%} {r['sharpe']:>8.2f} "
              f"{len(sell_trades):>6} {r['win_rate']:>8.0%}")

    # 详细交易记录
    print("\n" + "=" * 100)
    print("详细交易记录")
    print("=" * 100)

    for r in all_results:
        trades = r['trades']
        if len(trades) > 0:
            print(f"\n--- {r['symbol']} ({len([t for t in trades if t['type'] == 'SELL'])}笔交易) ---")
            print(f"{'买入日期':<12} {'买入价':>10} {'情绪':>8} → {'卖出日期':<12} {'卖出价':>10} {'原因':<12} {'收益':>10}")
            print("-" * 85)

            buy_trade = None
            for t in trades:
                if t['type'] == 'BUY':
                    buy_trade = t
                else:
                    if buy_trade:
                        buy_date = buy_trade['date'].strftime('%Y-%m-%d') if hasattr(buy_trade['date'], 'strftime') else str(buy_trade['date'])[:10]
                        sell_date = t['date'].strftime('%Y-%m-%d') if hasattr(t['date'], 'strftime') else str(t['date'])[:10]
                        ret = t['trade_return'] if t['trade_return'] else 0
                        print(f"{buy_date:<12} ${buy_trade['price']:>9.2f} {buy_trade['sentiment']:>8.1f} → "
                              f"{sell_date:<12} ${t['price']:>9.2f} {t['reason']:<12} {ret:>+10.2%}")
                    buy_trade = None

    # 最终资产
    print("\n" + "=" * 80)
    print("最终资产 (初始 $100,000)")
    print("=" * 80)

    print(f"\n{'股票':<8} {'最终资产':>15} {'净利润':>15}")
    print("-" * 45)

    total_final = 0
    for r in sorted(all_results, key=lambda x: x['final_value'], reverse=True):
        profit = r['final_value'] - INITIAL_CAPITAL
        total_final += r['final_value']
        print(f"{r['symbol']:<8} ${r['final_value']:>14,.0f} ${profit:>+14,.0f}")

    print("-" * 45)
    total_invested = INITIAL_CAPITAL * len(all_results)
    total_profit = total_final - total_invested
    print(f"{'合计':<8} ${total_final:>14,.0f} ${total_profit:>+14,.0f}")

    print("\n" + "=" * 80)
    print(f"完成! 所有文件保存在 {output_dir}/ 目录")
    print("=" * 80)


if __name__ == "__main__":
    main()
