"""
卖出策略分析
=====================================
分析当前卖出策略的问题:
1. 2022年熊市为什么没有触发卖出?
2. 卖出阈值过高导致错过卖出时机
3. 探索改进方案
"""

import sys
sys.path.insert(0, '/Users/sc2025/Desktop/Claude/Quantrade/src')

import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime
from data_loader import DataLoader

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "crypto_fear_greed_2",
    "user": "sc2025",
    "password": ""
}

def load_data(symbol):
    """加载价格和情绪数据"""
    # 价格
    loader = DataLoader(DB_CONFIG)
    price_df = loader.load_ohlcv(symbol, start_date="2020-01-01")
    loader.close()

    # 情绪
    conn = psycopg2.connect(**DB_CONFIG)
    query = """
        SELECT date, smoothed_index
        FROM fear_greed_index_s3
        WHERE symbol = %s
        ORDER BY date
    """
    sentiment_df = pd.read_sql(query, conn, params=(symbol,))
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'], utc=True)
    sentiment_df = sentiment_df.set_index('date')
    conn.close()

    # 合并
    df = price_df.copy()
    df['sentiment'] = sentiment_df['smoothed_index'].reindex(df.index)
    df['MA50'] = df['Close'].rolling(50).mean()
    df = df.dropna()

    return df


def analyze_2022_sell_conditions(symbol):
    """分析2022年为什么没有触发卖出"""
    df = load_data(symbol)
    df_2022 = df['2022-01-01':'2022-12-31']

    print(f"\n{'='*70}")
    print(f"  {symbol} - 2022年卖出条件分析")
    print(f"{'='*70}")

    # 基本统计
    print(f"\n  情绪统计:")
    print(f"    最高: {df_2022['sentiment'].max():.1f}")
    print(f"    最低: {df_2022['sentiment'].min():.1f}")
    print(f"    平均: {df_2022['sentiment'].mean():.1f}")

    # 检查各阈值触发次数
    thresholds = [15, 20, 25, 30, 40, 50, 60]

    print(f"\n  OR条件触发分析 (sent > threshold):")
    for t in thresholds:
        days = (df_2022['sentiment'] > t).sum()
        print(f"    sent > {t}: {days}天 ({days/len(df_2022)*100:.1f}%)")

    print(f"\n  AND条件触发分析 (sent > threshold 且 price < MA50):")
    for t in thresholds:
        and_cond = (df_2022['sentiment'] > t) & (df_2022['Close'] < df_2022['MA50'])
        days = and_cond.sum()
        print(f"    sent > {t} & P<MA50: {days}天 ({days/len(df_2022)*100:.1f}%)")

    # 价格与MA50的关系
    below_ma50 = (df_2022['Close'] < df_2022['MA50']).sum()
    print(f"\n  价格与MA50:")
    print(f"    Price < MA50: {below_ma50}天 ({below_ma50/len(df_2022)*100:.1f}%)")

    # 最大回撤
    peak = df_2022['Close'].expanding().max()
    drawdown = (df_2022['Close'] - peak) / peak * 100
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()

    print(f"\n  最大回撤:")
    print(f"    {max_dd:.1f}% (发生在 {max_dd_date.strftime('%Y-%m-%d')})")

    # 如果持仓，何时应该卖出?
    print(f"\n  回撤超过阈值的天数:")
    for dd_thresh in [-10, -20, -30, -40, -50]:
        days = (drawdown < dd_thresh).sum()
        first_day = drawdown[drawdown < dd_thresh].index[0] if days > 0 else None
        if first_day:
            print(f"    回撤 < {dd_thresh}%: {days}天 (首次: {first_day.strftime('%Y-%m-%d')})")
        else:
            print(f"    回撤 < {dd_thresh}%: 0天")

    return df_2022


def analyze_missed_sells(symbol):
    """分析错过的卖出机会"""
    df = load_data(symbol)

    print(f"\n{'='*70}")
    print(f"  {symbol} - 错过的卖出机会分析")
    print(f"{'='*70}")

    # 找出情绪高点
    df['sent_peak'] = df['sentiment'].rolling(20).max()
    df['is_local_peak'] = df['sentiment'] == df['sent_peak']

    # 情绪高点后的价格变化
    peaks = df[df['is_local_peak'] & (df['sentiment'] > 15)].copy()

    print(f"\n  情绪高点 (sent > 15) 后20天价格变化:")
    print(f"  {'日期':<12} {'情绪':>8} {'当日价格':>10} {'20日后':>10} {'变化':>10}")
    print(f"  {'-'*55}")

    count = 0
    for idx, row in peaks.iterrows():
        if count >= 15:
            break
        future_idx = df.index.get_loc(idx) + 20
        if future_idx < len(df):
            future_price = df.iloc[future_idx]['Close']
            change = (future_price - row['Close']) / row['Close'] * 100
            if abs(change) > 5:  # 只显示变化大于5%的
                print(f"  {idx.strftime('%Y-%m-%d'):<12} {row['sentiment']:>8.1f} ${row['Close']:>9.2f} ${future_price:>9.2f} {change:>+9.1f}%")
                count += 1


def propose_sell_improvements():
    """提出卖出策略改进建议"""
    print(f"\n{'='*70}")
    print(f"  卖出策略改进建议")
    print(f"{'='*70}")

    print("""
  【问题1】2022年熊市无法触发卖出
  ─────────────────────────────────
  原因:
    - OR阈值过高 (如60)，情绪从未达到
    - AND条件需要 price < MA50，但在快速下跌中可能不满足

  改进方案:
    A. 硬性止损: 不管情绪，亏损达到-X%就卖出
       - 高波动股: -15%
       - 低波动股: -10%

    B. 移动止盈: 盈利达到+X%后，回撤Y%就卖出
       - 盈利>20%后，回撤>10%卖出
       - 盈利>50%后，回撤>15%卖出

    C. 时间止损: 持仓超过N天且未盈利，考虑减仓
       - 持仓>60天且亏损>10% → 减仓50%

  【问题2】卖出阈值过高错过时机
  ─────────────────────────────────
  原因:
    - OR>60 对NVDA可能一年只触发几次
    - 等待高点过程中利润回吐

  改进方案:
    A. 分批卖出 (类似分批买入):
       - sent > 15 → 卖出25%
       - sent > 25 → 卖出25%
       - sent > 40 → 卖出25%
       - sent > 55 → 卖出25%

    B. 动态阈值: 根据持仓时间调整
       - 持仓<30天: 使用原阈值
       - 持仓30-60天: 阈值-5
       - 持仓>60天: 阈值-10

    C. 利润保护: 达到目标利润后降低卖出阈值
       - 盈利<20%: 原阈值
       - 盈利20-50%: 阈值-10
       - 盈利>50%: 阈值-15

  【问题3】AND条件的局限性
  ─────────────────────────────────
  原因:
    - 需要同时满足 sent>AND阈值 且 price<MA50
    - 在震荡市中可能两个条件不同时满足

  改进方案:
    A. 放宽AND条件:
       - 原: sent > AND 且 price < MA50
       - 新: sent > AND 且 price < MA20 (更敏感)

    B. 增加价格动量条件:
       - sent > AND 且 (price < MA50 或 5日跌幅 > 5%)

    C. 取消AND条件，只用OR:
       - 简化逻辑，只看情绪
       - 但需要降低OR阈值
    """)


def main():
    symbols = ["NVDA", "TSLA", "META", "AAPL", "MSFT", "GOOGL", "AMZN"]

    print("="*70)
    print("  卖出策略问题分析")
    print("="*70)

    # 分析2022年各股票的卖出条件
    for symbol in ["NVDA", "TSLA", "META"]:  # 选几个代表性的
        analyze_2022_sell_conditions(symbol)

    # 分析错过的卖出机会
    for symbol in ["NVDA"]:
        analyze_missed_sells(symbol)

    # 提出改进建议
    propose_sell_improvements()


if __name__ == "__main__":
    main()
