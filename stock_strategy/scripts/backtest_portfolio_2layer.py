#!/usr/bin/env python3
"""
两层仓位管理 + 动态复利组合回测
测试期: 2020年（Window5测试期）
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime
from data_loader import DataLoader

db_config = {
    'host': 'localhost',
    'database': 'crypto_fear_greed_2',
    'user': 'sc2025',
    'password': '',
    'port': 5432
}

def load_sentiment_index(symbol, smoothing=3):
    """加载情绪指数"""
    conn = psycopg2.connect(**db_config)

    if smoothing == 3:
        # 直接从S3表加载
        query = f"""
            SELECT date, smoothed_index
            FROM fear_greed_index_s3
            WHERE symbol = '{symbol}'
              AND date >= '2016-01-01'
              AND date <= '2020-12-31'
            ORDER BY date
        """
        df = pd.read_sql(query, conn, parse_dates=['date'])
        conn.close()
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df.set_index('date', inplace=True)
        return df
    else:
        # S5需要从原始数据计算
        query = f"""
            SELECT date, raw_index
            FROM fear_greed_index
            WHERE symbol = '{symbol}'
              AND date >= '2016-01-01'
              AND date <= '2020-12-31'
            ORDER BY date
        """
        df = pd.read_sql(query, conn, parse_dates=['date'])
        conn.close()
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df.set_index('date', inplace=True)

        # 计算S5平滑 (alpha = 1/5 = 0.2)
        df['smoothed_index'] = df['raw_index'].ewm(alpha=0.2, adjust=False).mean()
        df = df[['smoothed_index']]
        return df

class StockPool:
    """单个股票的资金池"""
    def __init__(self, symbol, initial_capital, buy_pct, params, smoothing):
        self.symbol = symbol
        self.total_capital = initial_capital
        self.cash = initial_capital
        self.shares = 0
        self.entry_price = 0
        self.entry_date = None
        self.buy_pct = buy_pct
        self.params = params
        self.smoothing = smoothing
        self.trades = []

    def can_buy(self):
        """检查是否可以买入"""
        return self.shares == 0 and self.cash > 1000

    def buy(self, date, price, sentiment_idx):
        """执行买入"""
        if not self.can_buy():
            return False

        # 计算买入金额
        buy_amount = self.cash * self.buy_pct

        # 计算股数（扣除手续费0.2%和滑点0.2%）
        shares = int(buy_amount / (price * 1.002))

        if shares == 0:
            return False

        actual_cost = shares * price * 1.002

        # 更新状态
        self.cash -= actual_cost
        self.shares = shares
        self.entry_price = price * 1.002
        self.entry_date = date

        return True

    def sell(self, date, price, reason, sentiment_idx):
        """执行卖出"""
        if self.shares == 0:
            return False

        # 计算卖出收入（扣除手续费0.2%和滑点0.2%）
        revenue = self.shares * price * 0.998

        # 计算盈亏
        profit = revenue - (self.shares * self.entry_price)
        profit_pct = (profit / (self.shares * self.entry_price)) * 100

        # 记录交易
        self.trades.append({
            'symbol': self.symbol,
            'entry_date': self.entry_date,
            'exit_date': date,
            'entry_price': self.entry_price,
            'exit_price': price * 0.998,
            'shares': self.shares,
            'profit': profit,
            'profit_pct': profit_pct,
            'exit_reason': reason,
            'holding_days': (date - self.entry_date).days
        })

        # 更新资金池
        self.cash += revenue
        self.total_capital = self.cash  # 更新总资金池
        self.shares = 0
        self.entry_price = 0
        self.entry_date = None

        return True

    def get_value(self, current_price):
        """获取当前市值"""
        position_value = self.shares * current_price if self.shares > 0 else 0
        return self.cash + position_value

class PortfolioBacktester:
    """组合回测器 - 两层仓位管理"""
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital

        # 股票配置（第一层：战略层）
        self.allocations = {
            'MSFT':  0.30,  # 30%
            'GOOGL': 0.25,  # 25%
            'NVDA':  0.20,  # 20%
            'TSLA':  0.15,  # 15%
            'AAPL':  0.10,  # 10%
        }

        # 买入比例（第二层：战术层）
        self.buy_percentages = {
            'MSFT':  0.60,  # A级，60%
            'GOOGL': 0.50,  # B级，50%
            'NVDA':  0.40,  # C级，40%
            'TSLA':  0.40,  # C级，40%
            'AAPL':  0.30,  # C级最弱，30%
        }

        # 参数配置
        self.params = {
            'MSFT':  {'buy': -5,  'and': 15, 'or': 40, 'smoothing': 3},
            'GOOGL': {'buy': -5,  'and': 10, 'or': 40, 'smoothing': 5},
            'NVDA':  {'buy': 5,   'and': 10, 'or': 50, 'smoothing': 3},
            'TSLA':  {'buy': -15, 'and': 10, 'or': 35, 'smoothing': 5},
            'AAPL':  {'buy': -10, 'and': 25, 'or': 50, 'smoothing': 3},
        }

        # 初始化各股票资金池
        self.pools = {}
        for symbol, allocation in self.allocations.items():
            initial_pool = initial_capital * allocation
            buy_pct = self.buy_percentages[symbol]
            params = self.params[symbol]

            self.pools[symbol] = StockPool(
                symbol=symbol,
                initial_capital=initial_pool,
                buy_pct=buy_pct,
                params=params,
                smoothing=params['smoothing']
            )

    def run_backtest(self, test_start='2020-01-01', test_end='2020-12-31'):
        """运行回测"""
        print("="*80)
        print(f"组合回测：两层仓位管理 + 动态复利")
        print(f"测试期: {test_start} ~ {test_end}")
        print("="*80 + "\n")

        print("初始配置:")
        print(f"  总资金: ${self.initial_capital:,.0f}\n")

        for symbol, pool in self.pools.items():
            print(f"  {symbol}:")
            print(f"    资金池: ${pool.total_capital:,.0f} ({self.allocations[symbol]*100:.0f}%)")
            print(f"    买入比例: {pool.buy_pct*100:.0f}%")
            print(f"    首次买入约: ${pool.total_capital * pool.buy_pct:,.0f}")
            print(f"    参数: buy<{pool.params['buy']}, and>{pool.params['and']}, or>{pool.params['or']} (S{pool.params['smoothing']})")

        # 加载数据
        print("\n加载数据...")
        loader = DataLoader(db_config)

        data = {}
        for symbol in self.pools.keys():
            pool = self.pools[symbol]

            # 加载价格数据
            prices = loader.load_ohlcv(symbol, '2016-01-01', '2020-12-31')

            # 加载情绪指数
            sentiment = load_sentiment_index(symbol, pool.smoothing)

            # 对齐日期
            common_dates = prices.index.intersection(sentiment.index)
            prices = prices.loc[common_dates]
            sentiment = sentiment.loc[common_dates]

            # 只取测试期
            prices_test = prices.loc[test_start:test_end]
            sentiment_test = sentiment.loc[test_start:test_end]

            # 计算MA50
            ma50 = prices['Close'].rolling(50).mean()
            ma50_test = ma50.loc[test_start:test_end]

            data[symbol] = {
                'prices': prices_test,
                'sentiment': sentiment_test,
                'ma50': ma50_test
            }

            print(f"  {symbol}: {len(prices_test)} 个交易日")

        # 获取所有日期（取最长的）
        all_dates = None
        for symbol, d in data.items():
            if all_dates is None or len(d['prices']) > len(all_dates):
                all_dates = d['prices'].index

        # 逐日回测
        print(f"\n开始回测 ({len(all_dates)} 个交易日)...")

        portfolio_values = []

        for i, date in enumerate(all_dates):
            # 计算当前组合总价值
            total_value = 0

            for symbol, pool in self.pools.items():
                if date not in data[symbol]['prices'].index:
                    # 该股票当天无数据，用前一个价格
                    current_price = data[symbol]['prices']['Close'].iloc[-1] if i > 0 else pool.total_capital
                else:
                    current_price = data[symbol]['prices'].loc[date, 'Close']

                total_value += pool.get_value(current_price)

            portfolio_values.append({
                'date': date,
                'total_value': total_value
            })

            # 检查每只股票的买入/卖出信号
            for symbol, pool in self.pools.items():
                if date not in data[symbol]['prices'].index:
                    continue

                price = data[symbol]['prices'].loc[date, 'Close']

                if date not in data[symbol]['sentiment'].index:
                    continue

                idx = data[symbol]['sentiment'].loc[date, 'smoothed_index']

                if date not in data[symbol]['ma50'].index:
                    continue

                ma50 = data[symbol]['ma50'].loc[date]

                params = pool.params

                # 买入逻辑
                if pool.can_buy() and idx < params['buy']:
                    pool.buy(date, price, idx)

                # 卖出逻辑
                if pool.shares > 0:
                    sell_signal = False
                    exit_reason = None

                    # OR条件
                    if idx > params['or']:
                        sell_signal = True
                        exit_reason = 'OR'
                    # AND条件
                    elif idx > params['and'] and price < ma50:
                        sell_signal = True
                        exit_reason = 'AND'

                    if sell_signal:
                        pool.sell(date, price, exit_reason, idx)

        # 期末强制平仓
        final_date = all_dates[-1]
        for symbol, pool in self.pools.items():
            if pool.shares > 0:
                final_price = data[symbol]['prices'].loc[final_date, 'Close']
                pool.sell(final_date, final_price, 'EOD', 0)

        # 汇总结果
        portfolio_df = pd.DataFrame(portfolio_values)
        portfolio_df.set_index('date', inplace=True)

        # 计算指标
        final_value = portfolio_df['total_value'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100

        # 计算Sharpe
        portfolio_df['returns'] = portfolio_df['total_value'].pct_change()
        returns = portfolio_df['returns'].dropna()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

        # 计算回撤
        cummax = portfolio_df['total_value'].cummax()
        drawdown = (portfolio_df['total_value'] - cummax) / cummax * 100
        max_drawdown = drawdown.min()

        # 汇总所有交易
        all_trades = []
        for symbol, pool in self.pools.items():
            all_trades.extend(pool.trades)

        trades_df = pd.DataFrame(all_trades)

        if len(trades_df) > 0:
            win_trades = trades_df[trades_df['profit'] > 0]
            win_rate = len(win_trades) / len(trades_df)
        else:
            win_rate = 0

        # 打印结果
        print("\n" + "="*80)
        print("回测结果")
        print("="*80 + "\n")

        print(f"初始资金: ${self.initial_capital:,.2f}")
        print(f"最终资金: ${final_value:,.2f}")
        print(f"总收益: {total_return:.2f}%")
        print(f"Sharpe比率: {sharpe:.4f}")
        print(f"最大回撤: {max_drawdown:.2f}%")
        print(f"总交易次数: {len(trades_df)}")
        print(f"胜率: {win_rate*100:.1f}%")

        # 各股票表现
        print("\n" + "="*80)
        print("各股票表现")
        print("="*80 + "\n")

        for symbol, pool in self.pools.items():
            stock_trades = [t for t in all_trades if t['symbol'] == symbol]

            if len(stock_trades) > 0:
                stock_df = pd.DataFrame(stock_trades)
                stock_profit = stock_df['profit'].sum()
                stock_win_rate = len(stock_df[stock_df['profit'] > 0]) / len(stock_df)
                avg_profit_pct = stock_df['profit_pct'].mean()
            else:
                stock_profit = 0
                stock_win_rate = 0
                avg_profit_pct = 0

            final_pool = pool.total_capital
            pool_return = (final_pool - (self.initial_capital * self.allocations[symbol])) / (self.initial_capital * self.allocations[symbol]) * 100

            print(f"{symbol}:")
            print(f"  初始资金池: ${self.initial_capital * self.allocations[symbol]:,.0f}")
            print(f"  最终资金池: ${final_pool:,.2f}")
            print(f"  资金池收益: {pool_return:+.2f}%")
            print(f"  交易次数: {len(stock_trades)}")
            print(f"  胜率: {stock_win_rate*100:.1f}%" if len(stock_trades) > 0 else "  胜率: N/A")
            print(f"  平均单笔收益: {avg_profit_pct:.2f}%" if len(stock_trades) > 0 else "  平均单笔收益: N/A")
            print()

        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存组合价值
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        portfolio_file = os.path.join(output_dir, f'portfolio_2layer_{timestamp}.csv')
        portfolio_df.to_csv(portfolio_file)

        # 保存交易记录
        if len(trades_df) > 0:
            trades_file = os.path.join(output_dir, f'trades_2layer_{timestamp}.csv')
            trades_df.to_csv(trades_file, index=False)
            print(f"✅ 交易记录已保存: {trades_file}")

        print(f"✅ 组合价值已保存: {portfolio_file}")

        return {
            'total_return': total_return,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(trades_df),
            'final_value': final_value,
            'portfolio_df': portfolio_df,
            'trades_df': trades_df
        }

def main():
    # 运行回测 - 10年完整验证
    backtester = PortfolioBacktester(initial_capital=100000)
    results = backtester.run_backtest(test_start='2010-01-01', test_end='2020-12-31')

    print("\n" + "="*80)
    print("💡 结论")
    print("="*80 + "\n")

    # 计算年化收益
    years = 11  # 2010-2020是11年
    annualized_return = ((1 + results['total_return']/100) ** (1/years) - 1) * 100

    print(f"两层仓位管理 + 动态复利策略（2010-2020年，11年）表现:")
    print(f"  总收益: {results['total_return']:.2f}%")
    print(f"  年化收益: {annualized_return:.2f}%")
    print(f"  Sharpe: {results['sharpe']:.4f}")
    print(f"  最大回撤: {results['max_drawdown']:.2f}%")
    print(f"  胜率: {results['win_rate']*100:.1f}%")
    print(f"  总交易次数: {results['num_trades']}")

    if results['sharpe'] > 1.0 and annualized_return > 15:
        print("\n✅ 策略可行！表现优秀")
    elif results['sharpe'] > 0.8 and annualized_return > 10:
        print("\n⭐ 策略可行！表现良好")
    else:
        print("\n⚠️ 策略表现一般，需要优化")

if __name__ == '__main__':
    main()
