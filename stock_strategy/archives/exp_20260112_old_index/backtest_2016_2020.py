"""
2016-2020 年度独立回测 - 样本外验证
使用 2021-2025 实验得出的最优参数
验证策略在不同市场环境下的稳健性
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
POSITION_PCT = 0.8  # 全部使用动态仓位
MA_PERIOD = 50

# 七姐妹股票
MAG7_STOCKS = ['NVDA', 'TSLA', 'GOOGL', 'META', 'MSFT', 'AAPL', 'AMZN']

# 2021-2025 实验得出的最优参数 (全部用动态仓位)
OPTIMAL_CONFIG = {
    'NVDA': {'buy_th': -10, 'sell_mode': 'and_ma', 'sell_th': 10},
    'TSLA': {'buy_th': -10, 'sell_mode': 'and_ma', 'sell_th': 10},
    'GOOGL': {'buy_th': -5, 'sell_mode': 'threshold', 'sell_th': 30},
    'META': {'buy_th': -10, 'sell_mode': 'and_ma', 'sell_th': 5},
    'MSFT': {'buy_th': -10, 'sell_mode': 'threshold', 'sell_th': 30},
    'AAPL': {'buy_th': -10, 'sell_mode': 'threshold', 'sell_th': 30},
    'AMZN': {'buy_th': -10, 'sell_mode': 'and_ma', 'sell_th': 5},
}

# 测试年份
TEST_YEARS = [2016, 2017, 2018, 2019, 2020]


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


def run_yearly_backtest(stock_data, sentiment_data, config, year):
    """运行单年度回测"""
    buy_th = config['buy_th']
    sell_mode = config['sell_mode']
    sell_th = config['sell_th']

    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    # 需要额外的lookback数据计算MA
    lookback_start = pd.to_datetime(start_date, utc=True) - pd.Timedelta(days=MA_PERIOD * 2)
    end_dt = pd.to_datetime(end_date, utc=True)
    mask = (stock_data.index >= lookback_start) & (stock_data.index <= end_dt)
    price_data = stock_data[mask].copy()

    if len(price_data) < MA_PERIOD:
        return None

    price_data['MA'] = price_data['Close'].rolling(window=MA_PERIOD).mean()

    # 只取测试年份的数据
    start_dt = pd.to_datetime(start_date, utc=True)
    test_mask = price_data.index >= start_dt
    price_data = price_data[test_mask].copy()

    if len(price_data) == 0 or sentiment_data is None:
        return None

    # 对齐情绪数据
    sent_data = sentiment_data.reindex(price_data.index)

    # 初始化
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
                proceeds = shares * sell_price
                cash += proceeds
                trade_return = (sell_price - entry_price) / entry_price
                trades.append({
                    'type': 'SELL',
                    'date': date,
                    'price': price,
                    'sentiment': sentiment,
                    'ma': ma,
                    'reason': exit_reason,
                    'shares': shares,
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'entry_sentiment': entry_sentiment,
                    'trade_return': trade_return
                })
                shares = 0
                entry_price = 0

        # 买入逻辑 - 动态仓位
        elif shares == 0 and sentiment < buy_th:
            current_total_for_position = cash + shares * price
            position_value = current_total_for_position * POSITION_PCT

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
                        'type': 'BUY',
                        'date': date,
                        'price': price,
                        'sentiment': sentiment,
                        'ma': ma,
                        'reason': f'sentiment<{buy_th}',
                        'shares': shares,
                        'entry_date': None,
                        'entry_price': None,
                        'entry_sentiment': None,
                        'trade_return': None
                    })
                else:
                    shares = 0

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
        trade_return = (sell_price - entry_price) / entry_price
        trades.append({
            'type': 'SELL',
            'date': price_data.index[-1],
            'price': final_price,
            'sentiment': final_sentiment,
            'ma': final_ma,
            'reason': 'year_end',
            'shares': shares,
            'entry_date': entry_date,
            'entry_price': entry_price,
            'entry_sentiment': entry_sentiment,
            'trade_return': trade_return
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
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'trades': trades,
        'trade_count': len(sell_trades),
        'win_rate': win_rate,
        'final_value': final_value,
        'portfolio_series': portfolio_series,
        'price_data': price_data
    }


def create_visualization(symbol, all_results, output_dir):
    """创建可视化图表"""
    fig, axes = plt.subplots(len(TEST_YEARS), 1, figsize=(16, 4 * len(TEST_YEARS)))

    config = OPTIMAL_CONFIG[symbol]
    config_str = f"buy<{config['buy_th']}, "
    if config['sell_mode'] == 'threshold':
        config_str += f"sell>{config['sell_th']}"
    else:
        config_str += f"sell>{config['sell_th']}&<MA"

    for idx, year in enumerate(TEST_YEARS):
        ax = axes[idx] if len(TEST_YEARS) > 1 else axes

        if year not in all_results or all_results[year] is None:
            ax.text(0.5, 0.5, f'{year}: 无数据', ha='center', va='center', fontsize=14)
            ax.set_title(f'{symbol} {year}')
            continue

        result = all_results[year]
        price_data = result['price_data']
        trades = result['trades']

        # 绘制价格和MA
        ax.plot(price_data.index, price_data['Close'], 'b-', alpha=0.7, label='Price')
        ax.plot(price_data.index, price_data['MA'], 'orange', alpha=0.5, label='MA50')

        # 绘制买卖点
        for trade in trades:
            if trade['type'] == 'BUY':
                ax.scatter(trade['date'], trade['price'], color='green', marker='^',
                          s=150, zorder=5, edgecolors='darkgreen', linewidths=1.5)
            else:
                color = 'red' if trade['reason'] != 'year_end' else 'blue'
                ax.scatter(trade['date'], trade['price'], color=color, marker='v',
                          s=150, zorder=5, edgecolors='dark'+color if color != 'blue' else 'darkblue', linewidths=1.5)

        # 标题和标签
        ret = result['total_return']
        dd = result['max_drawdown']
        sharpe = result['sharpe']
        trades_count = result['trade_count']

        ax.set_title(f'{symbol} {year}: 收益 {ret:+.2%} | 回撤 {dd:.2%} | 夏普 {sharpe:.2f} | 交易 {trades_count}次',
                    fontsize=12)
        ax.set_ylabel('Price')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())

    plt.suptitle(f'{symbol} 2016-2020 年度独立回测 ({config_str}, 动态仓位)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{symbol}_2016_2020.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 100)
    print("2016-2020 年度独立回测 - 样本外验证")
    print("使用 2021-2025 实验得出的最优参数，全部动态仓位")
    print("=" * 100)

    loader = DataLoader(db_config)

    # 创建输出目录
    output_dir = 'backtest_2016_2020_results'
    os.makedirs(output_dir, exist_ok=True)

    all_trades = []
    summary_data = []

    # 汇总表格 - 年度收益
    yearly_returns = {year: {} for year in TEST_YEARS}
    yearly_sharpe = {year: {} for year in TEST_YEARS}
    yearly_drawdown = {year: {} for year in TEST_YEARS}

    for symbol in MAG7_STOCKS:
        print(f"\n{'='*80}")
        print(f"测试 {symbol}")
        print("=" * 80)

        config = OPTIMAL_CONFIG[symbol]
        config_str = f"buy<{config['buy_th']}, "
        if config['sell_mode'] == 'threshold':
            config_str += f"sell>{config['sell_th']}"
        else:
            config_str += f"sell>{config['sell_th']}&<MA"
        print(f"配置: {config_str}, 动态仓位")

        stock_data = loader.load_ohlcv(symbol, start_date="2014-01-01", end_date="2020-12-31")
        sentiment_data = load_sentiment_data(symbol)

        if stock_data is None or len(stock_data) == 0:
            print(f"  {symbol} 无价格数据")
            continue

        if sentiment_data is None or len(sentiment_data) == 0:
            print(f"  {symbol} 无情绪数据")
            continue

        # 检查情绪数据范围
        sent_start = sentiment_data.index.min()
        sent_end = sentiment_data.index.max()
        print(f"  情绪数据范围: {sent_start.strftime('%Y-%m-%d')} ~ {sent_end.strftime('%Y-%m-%d')}")

        all_results = {}
        symbol_returns = []
        symbol_sharpe = []
        symbol_drawdown = []

        print(f"\n  {'年份':<6} {'收益':>10} {'回撤':>10} {'夏普':>8} {'交易':>6} {'胜率':>8}")
        print(f"  {'-'*55}")

        for year in TEST_YEARS:
            result = run_yearly_backtest(stock_data, sentiment_data, config, year)
            all_results[year] = result

            if result:
                yearly_returns[year][symbol] = result['total_return']
                yearly_sharpe[year][symbol] = result['sharpe']
                yearly_drawdown[year][symbol] = result['max_drawdown']

                symbol_returns.append(result['total_return'])
                symbol_sharpe.append(result['sharpe'])
                symbol_drawdown.append(result['max_drawdown'])

                print(f"  {year:<6} {result['total_return']:>+10.2%} {result['max_drawdown']:>10.2%} "
                      f"{result['sharpe']:>8.2f} {result['trade_count']:>6} {result['win_rate']:>8.0%}")

                # 保存交易记录
                for trade in result['trades']:
                    all_trades.append({
                        'symbol': symbol,
                        'year': year,
                        'type': trade['type'],
                        'date': trade['date'],
                        'price': trade['price'],
                        'sentiment': trade['sentiment'],
                        'ma': trade['ma'],
                        'reason': trade['reason'],
                        'shares': trade['shares'],
                        'entry_date': trade['entry_date'],
                        'entry_price': trade['entry_price'],
                        'trade_return': trade['trade_return']
                    })
            else:
                print(f"  {year:<6} {'无数据':>10}")

        # 汇总
        if symbol_returns:
            avg_return = np.mean(symbol_returns)
            cumulative = np.prod([1 + r for r in symbol_returns]) - 1
            avg_sharpe = np.mean(symbol_sharpe)
            avg_drawdown = np.mean(symbol_drawdown)

            print(f"  {'-'*55}")
            print(f"  {'平均':<6} {avg_return:>+10.2%} {avg_drawdown:>10.2%} {avg_sharpe:>8.2f}")
            print(f"  {'5年累计':<6} {cumulative:>+10.2%}")

            summary_data.append({
                'symbol': symbol,
                'config': config_str,
                'avg_return': avg_return,
                'cumulative': cumulative,
                'avg_drawdown': avg_drawdown,
                'avg_sharpe': avg_sharpe
            })

        # 生成可视化
        create_visualization(symbol, all_results, output_dir)
        print(f"  图表已保存: {output_dir}/{symbol}_2016_2020.png")

    loader.close()

    # 保存交易记录
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_df.to_csv(f'{output_dir}/all_trades.csv', index=False)
        print(f"\n交易记录已保存: {output_dir}/all_trades.csv")

    # 保存汇总
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f'{output_dir}/summary.csv', index=False)

    # 打印汇总表格
    print("\n" + "=" * 100)
    print("2016-2020 vs 2021-2025 对比")
    print("=" * 100)

    # 年度收益表
    print(f"\n{'股票':<8}", end="")
    for year in TEST_YEARS:
        print(f"{year:>10}", end="")
    print(f"{'5年累计':>12} {'平均':>10}")
    print("-" * 80)

    for symbol in MAG7_STOCKS:
        print(f"{symbol:<8}", end="")
        returns = []
        for year in TEST_YEARS:
            if symbol in yearly_returns[year]:
                ret = yearly_returns[year][symbol]
                returns.append(ret)
                print(f"{ret:>+10.2%}", end="")
            else:
                print(f"{'N/A':>10}", end="")

        if returns:
            cumulative = np.prod([1 + r for r in returns]) - 1
            avg = np.mean(returns)
            print(f"{cumulative:>+12.2%} {avg:>+10.2%}")
        else:
            print()

    # 与2021-2025对比
    print("\n" + "=" * 100)
    print("样本外验证总结")
    print("=" * 100)

    # 2021-2025 数据 (从之前实验)
    results_2021_2025 = {
        'NVDA': {'cumulative': 7.7277, 'avg_return': 0.6853},
        'TSLA': {'cumulative': 2.2733, 'avg_return': 0.4746},
        'GOOGL': {'cumulative': 2.0895, 'avg_return': 0.2956},
        'META': {'cumulative': 1.4337, 'avg_return': 0.3126},
        'MSFT': {'cumulative': 0.9917, 'avg_return': 0.1695},
        'AAPL': {'cumulative': 0.9101, 'avg_return': 0.1986},
        'AMZN': {'cumulative': 0.4417, 'avg_return': 0.1463},
    }

    print(f"\n{'股票':<8} {'2016-2020累计':>15} {'2021-2025累计':>15} {'差异':>12} {'验证结果':>12}")
    print("-" * 70)

    valid_count = 0
    for item in summary_data:
        symbol = item['symbol']
        cum_1620 = item['cumulative']
        cum_2125 = results_2021_2025[symbol]['cumulative']
        diff = cum_1620 - cum_2125

        # 判断验证结果
        if cum_1620 > 0.3:  # 5年累计超过30%算有效
            result = "有效"
            valid_count += 1
        elif cum_1620 > 0:
            result = "一般"
        else:
            result = "失效"

        print(f"{symbol:<8} {cum_1620:>+15.2%} {cum_2125:>+15.2%} {diff:>+12.2%} {result:>12}")

    print("-" * 70)
    print(f"策略验证通过率: {valid_count}/{len(summary_data)} = {valid_count/len(summary_data)*100:.0f}%")

    print("\n" + "=" * 100)
    print(f"完成! 所有文件保存在 {output_dir}/ 目录")
    print("=" * 100)


if __name__ == "__main__":
    main()
