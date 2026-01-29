# 情绪指数交易系统 - 技术文档

**给开发者/维护者的详细技术说明**

---

## 项目概述

基于市场情绪指数的量化交易系统，支持美股七强和加密货币 (BTC, ETH, SOL)。

**核心特点**:
- 情绪驱动的买卖决策 (非 ML 预测)
- 间隔分批买入策略 (v3.0)
- AND/OR 双条件卖出
- 网格搜索参数优化
- 季度更新机制

---

## 项目结构

```
sentiment-index-trading/
├── stock_strategy/          # 美股情绪策略实验
│   ├── exp_*/               # 各类实验目录
│   ├── visualize_*.py       # 可视化工具
│   └── archives/            # 历史存档
├── crypto_strategy/         # 加密货币情绪策略
│   ├── crypto_backtest.py   # 主回测脚本
│   ├── crypto_threshold_search.py  # 阈值搜索
│   └── crypto_train_mode_comparison.py  # 训练模式对比
├── production/              # 生产版本
│   ├── stock/               # 美股生产 (季度更新)
│   └── crypto/              # 加密货币生产 (季度更新)
├── docs/                    # 文档
│   ├── WORK_PROGRESS_SUMMARY.md
│   └── 2026_THRESHOLD_STAGED_RECOMMENDATION.md
├── README.md
└── CLAUDE.md                # 本文件
```

---

## 📋 目录

1. [项目概述](#项目概述)
2. [数据源](#数据源)
3. [核心策略](#核心策略)
4. [美股策略详解](#美股策略详解)
5. [加密货币策略详解](#加密货币策略详解)
6. [生产系统](#生产系统)
7. [实验记录](#实验记录)
8. [代码模块](#代码模块)
9. [扩展指南](#扩展指南)

---

## 数据源

### 美股情绪指数

**数据库**: PostgreSQL (`crypto_fear_greed_2`)

**表**: `fear_greed_index_s3`

| 字段 | 类型 | 说明 |
|------|------|------|
| `date` | DATE | 日期 |
| `symbol` | VARCHAR | 股票代码 |
| `fear_greed_index` | FLOAT | 情绪指数 (-100 ~ +100) |

**参数**:
- Smoothing = 3 (3日平滑)
- MoneyFlow = 13 (13日资金流)

**查询示例**:
```sql
SELECT date, fear_greed_index
FROM fear_greed_index_s3
WHERE symbol = 'NVDA'
ORDER BY date;
```

### 加密货币情绪指数

**表**: `yahoo_artemis_index`

| 字段 | 类型 | 说明 |
|------|------|------|
| `date` | DATE | 日期 |
| `symbol` | VARCHAR | 币种 (BTC, ETH, SOL) |
| `smoothed_index` | FLOAT | 平滑情绪指数 |

**表**: `yahoo_candles` (价格数据)

| 字段 | 类型 | 说明 |
|------|------|------|
| `open_time` | BIGINT | 时间戳 (ms) |
| `symbol` | VARCHAR | 币种 |
| `open_price` | DECIMAL | 开盘价 |
| `close_price` | DECIMAL | 收盘价 |
| `high_price` | DECIMAL | 最高价 |
| `low_price` | DECIMAL | 最低价 |
| `volume` | DECIMAL | 成交量 |

---

## 核心策略

### 策略演进

```
v1.0 原始策略        → 单次买入 + OR/AND卖出
v2.0 阈值分批策略    → sent<5/0/-5/-10, 每档25% (4批)
v3.0 间隔买入策略    → sent<0进入, 每7天买20%, 回升全买 ⭐当前生产版本
```

### v3.0 间隔分批买入

**核心机制**:
```
情绪 < 买入阈值 → 进入买入模式, 买入 20%
每隔 7 天 → 再买 20% (最多 5 批 = 100%)
情绪 >= 买入阈值 → 买入全部剩余, 退出买入模式
```

**优势**:
- 解决 v2.0 阈值分批在温和下跌中只触发 1-2 批的问题
- 资金利用率更高
- 不依赖固定阈值 (5/0/-5/-10)

**伪代码**:
```python
if not in_buy_mode and sentiment < buy_threshold:
    in_buy_mode = True
    buy_base = current_total_value
    buy(buy_base * 0.20)
    last_buy_date = today

if in_buy_mode:
    if sentiment >= buy_threshold:
        buy_all_remaining()
        in_buy_mode = False
    elif days_since(last_buy_date) >= 7:
        buy(buy_base * 0.20)
        last_buy_date = today
```

### AND/OR 双条件卖出

**OR 条件** (无条件卖出):
```python
if sentiment > OR_threshold:
    sell_all()
```

**AND 条件** (双重确认):
```python
if sentiment > AND_threshold and price < MA50:
    sell_all()
```

**设计理由**:
- OR: 情绪极端乐观时无条件退出
- AND: 情绪偏高 + 技术面走弱时退出 (避免假信号)

---

## 美股策略详解

### 标的资产

| 股票 | 配置比例 | 买入阈值 | 最优阈值 |
|------|---------|---------|---------|
| NVDA | 25% | 0 | 0 |
| TSLA | 15% | 0 | -15 |
| AAPL | 15% | 0 | 5 |
| GOOGL | 15% | 0 | -5 |
| MSFT | 10% | 0 | 0 |
| AMZN | 10% | 0 | 0 |
| META | - | 0 | 0 |

> **买入阈值**: 统一使用 `buy < 0`，最优阈值供参考

### 卖出参数 (2026-Q1)

训练期: 2022-2025 (4年), 使用 interval buy training

| 股票 | AND | OR | 训练收益 | 特征 |
|------|-----|-----|---------|------|
| NVDA | >5 | >50 | +517.9% | 敏感, 快进快出 |
| TSLA | >25 | >30 | +233.6% | AND宽松, OR敏感 |
| AAPL | >5 | >30 | +111.4% | 最敏感 |
| MSFT | >5 | >30 | +75.0% | 敏感, 快进快出 |
| GOOGL | >10 | >50 | +137.4% | 中等偏松 |
| AMZN | >10 | >30 | +88.3% | 中等 |
| META | >15 | >55 | +141.9% | 稳定 |

### Walk-Forward 回测结果

7只股票, 7个窗口 (2019-2025), 每窗口4年训练+1年测试:

| 股票 | 总收益 | 最差窗口 | 最优窗口 |
|------|--------|---------|---------|
| NVDA | +1510.7% | +323.0% | +573.4% |
| TSLA | +413.3% | +133.9% | +549.2% |
| AAPL | +310.0% | +118.7% | +331.4% |
| MSFT | +225.0% | +59.7% | +263.8% |
| GOOGL | +280.3% | +58.8% | +179.5% |
| AMZN | +215.1% | +10.1% | +276.9% |
| META | +122.0% | -1.8% | +128.0% |

**合计**: $3,776K (初始 $700K, 每股 $100K)

---

## 加密货币策略详解

### 标的资产

| 币种 | 买入阈值 | 训练周期 | 原因 |
|------|---------|---------|------|
| BTC | sent < 10 | 3年 | 情绪偏高 (均值 11.9), 需宽松阈值 |
| ETH | sent < -10 | 3年 | 选择性买入 |
| SOL | sent < -10 | 2年 | 避免 2021 百倍涨幅偏差 |

### 卖出参数范围

比美股更宽，适应高波动:

```python
AND_SELL_RANGE = [5, 10, 15, 20, 25]
OR_SELL_RANGE = [30, 50, 70, 90, 120, 150, 200]  # 美股仅 30-60
```

### 卖出参数 (2026-Q1)

| 币种 | AND | OR | 训练收益 | 训练期 |
|------|-----|-----|---------|--------|
| BTC | >20 | >50 | +352.3% | 2023-2025 (3yr) |
| ETH | >5 | >30 | +148.2% | 2023-2025 (3yr) |
| SOL | >15 | >30 | +104.4% | 2024-2025 (2yr) |

### 回测结果 (2023-2025)

| 币种 | v2.0 阈值分批 | v3.0 间隔分批 | 最优阈值 |
|------|-------------|-------------|---------|
| BTC | +50.1% ($150K) | **+344.3% ($444K)** | buy<10 |
| ETH | +60.8% ($161K) | **+90.3% ($190K)** | buy<-10 |
| SOL | +1000.3% ($1.10M) | **+1195.4% ($1.30M)** | buy<-10 |
| **合计** | **$1,411K** | **$1,930K** | |

**间隔分批 3:0 全胜**，多赚 $519K (+36.8%)

### 关键差异 (vs 美股)

| 项目 | 美股 | 加密货币 |
|------|------|---------|
| 买入阈值 | 统一 buy<0 | 逐币设定 |
| OR 范围 | 30-60 | 30-200 |
| 训练周期 | 统一 4 年 | BTC/ETH 3年, SOL 2年 |
| 交易单位 | 整数股 | 小数 (最小 0.0001) |

---

## 生产系统

### 目录结构

```
production/
├── stock/                   # 美股生产
│   ├── OPERATION_GUIDE.md   # 操作指南
│   ├── retrain_params.py    # 季度更新脚本
│   └── params_*.csv         # 参数文件
└── crypto/                  # 加密货币生产
    ├── OPERATION_GUIDE.md   # 操作指南
    ├── retrain_params.py    # 季度更新脚本
    └── params_*.csv         # 参数文件
```

### 美股季度更新

```bash
cd production/stock
python3 retrain_params.py              # 自动用最近4年
python3 retrain_params.py 2022 2026    # 或指定训练期
```

**输出**:
- `params_YYYYMMDD.csv`: 买入+卖出参数
- 控制台: 当前信号状态

### 加密货币季度更新

```bash
cd production/crypto
python3 retrain_params.py              # 自动按币种选训练期
python3 retrain_params.py 2023 2025    # 或指定训练期
```

### 季度更新计划

| 时间 | 美股训练期 | BTC/ETH 训练期 | SOL 训练期 |
|------|-----------|---------------|-----------|
| 2026-01 | 2022-2025 | 2023-2025 | 2024-2025 |
| 2026-04 | 2022-2026 Q1 | 2023-2026 Q1 | 2024-2026 Q1 |
| 2026-07 | 2022-2026 H1 | 2023-2026 H1 | 2024-2026 H1 |
| 2026-10 | 2023-2026 Q3 | 2024-2026 Q3 | 2025-2026 Q3 |

### 参数文件格式

`params_YYYYMMDD.csv`:
```csv
symbol,buy_threshold,optimal_buy,and_threshold,or_threshold,train_return,train_period
NVDA,0,0,5,50,517.9,2022-2025
TSLA,0,-15,25,30,233.6,2022-2025
...
```

---

## 实验记录

### 美股实验 (`stock_strategy/`)

| 目录 | 实验内容 | 结论 |
|------|---------|------|
| `exp_interval_buy/` | 间隔买入策略 (v3.0) | 确认为生产方案 |
| `exp_buy_strategy_staged_vs_interval/` | 阈值分批 vs 间隔分批 | 间隔分批 6:1 胜出 |
| `exp_sentiment_index_comparison/` | S3/S5/S3+VIX/S5+VIX 对比 | S3 最均衡 |
| `exp_fear_greed_mf26/` | MF26 vs MF13 对比 | MF13 更稳定 |
| `interval_train_experiment/` | 训练模式对比 | 训练须与测试一致 |

### 加密货币实验 (`crypto_strategy/`)

| 文件 | 实验内容 | 结论 |
|------|---------|------|
| `crypto_backtest.py` | 阈值 vs 间隔对比 | 间隔分批 3:0 胜出 |
| `crypto_threshold_search.py` | 买入阈值搜索 | BTC<10, ETH<-10, SOL<-10 |
| `crypto_train_mode_comparison.py` | 训练模式对比 | Interval +9.8% |

### 关键发现

1. **买入策略**: 间隔分批 (v3.0) 全面胜出
2. **情绪指数**: S3 (Smoothing=3, MF=13) 最均衡
3. **训练模式**: 必须与测试策略一致 (interval buy training)
4. **VIX 因子**: 对高 beta 有帮助，对低 beta 有害，不纳入
5. **SOL 训练期**: 用 2 年，排除 2021 百倍异常

---

## 代码模块

### 数据加载

```python
def load_price(symbol):
    """从 fear_greed_index_s3 加载价格数据"""
    conn = psycopg2.connect(**DB_CONFIG)
    query = """
        SELECT date, close_price::float as Close
        FROM fear_greed_index_s3
        WHERE symbol = %s
        ORDER BY date
    """
    df = pd.read_sql(query, conn, params=(symbol,))
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.set_index('date')
    conn.close()
    return df

def load_sentiment(symbol):
    """从 fear_greed_index_s3 加载情绪指数"""
    conn = psycopg2.connect(**DB_CONFIG)
    query = """
        SELECT date, fear_greed_index::float as sentiment
        FROM fear_greed_index_s3
        WHERE symbol = %s
        ORDER BY date
    """
    df = pd.read_sql(query, conn, params=(symbol,))
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.set_index('date')
    conn.close()
    return df
```

### 间隔买入逻辑

```python
def run_interval_buy(df, and_t, or_t, buy_threshold):
    """间隔分批买入 + AND/OR 卖出"""
    cash = INITIAL_CAPITAL
    position = 0
    in_buy_mode = False
    last_buy_date = None
    buy_base = None

    for i in range(len(df)):
        dt = df.index[i]
        price = df['Close'].iloc[i]
        sent = df['sentiment'].iloc[i]
        ma50 = df['MA50'].iloc[i]

        # 卖出检查
        if position > 0:
            if sent > or_t or (sent > and_t and price < ma50):
                cash += position * price * (1 - SLIPPAGE) * (1 - COMMISSION)
                position = 0
                in_buy_mode = False

        # 买入检查
        if not in_buy_mode and sent < buy_threshold:
            in_buy_mode = True
            buy_base = cash + position * price
            last_buy_date = None

        if in_buy_mode:
            should_buy = False
            buy_all = False

            if last_buy_date is None:
                should_buy = True
            elif sent >= buy_threshold:
                should_buy = True
                buy_all = True
            elif (dt - last_buy_date).days >= INTERVAL_DAYS:
                should_buy = True

            if should_buy:
                buy_price = price * (1 + SLIPPAGE) * (1 + COMMISSION)
                if buy_all:
                    shares = cash / buy_price
                else:
                    shares = buy_base * INTERVAL_BATCH_PCT / buy_price

                if shares > 0 and cash >= shares * buy_price:
                    position += shares
                    cash -= shares * buy_price
                    last_buy_date = dt

                if buy_all:
                    in_buy_mode = False

    return cash + position * df['Close'].iloc[-1]
```

### 网格搜索

```python
def grid_search(train_df, buy_threshold):
    """搜索最优 AND/OR 参数"""
    AND_RANGE = [5, 10, 15, 20, 25]
    OR_RANGE = [30, 40, 50, 60]

    best_return = -float('inf')
    best_params = None

    for and_t in AND_RANGE:
        for or_t in OR_RANGE:
            final_value = run_interval_buy(train_df, and_t, or_t, buy_threshold)
            ret = (final_value / INITIAL_CAPITAL - 1) * 100
            if ret > best_return:
                best_return = ret
                best_params = (and_t, or_t)

    return best_params, best_return
```

---

## 扩展指南

### 1. 添加新股票

1. 确认数据库中有该股票的情绪数据
2. 在 `production/stock/retrain_params.py` 的 `SYMBOLS` 列表中添加
3. 运行 `python3 retrain_params.py` 生成参数

### 2. 添加新加密货币

1. 确认 `yahoo_candles` 和 `yahoo_artemis_index` 中有数据
2. 在 `production/crypto/retrain_params.py` 中:
   - 添加到 `COIN_CONFIG` (设置买入阈值和训练周期)
   - 添加到 `COIN_WINDOWS` (设置 Walk-Forward 窗口)
3. 运行脚本生成参数

### 3. 调整买入参数

```python
# production/stock/retrain_params.py
INTERVAL_DAYS = 7        # 买入间隔天数
INTERVAL_BATCH_PCT = 0.20  # 每批买入比例
```

### 4. 调整卖出搜索范围

```python
# 美股
AND_SELL_RANGE = [5, 10, 15, 20, 25]
OR_SELL_RANGE = [30, 40, 50, 60]

# 加密货币 (更宽)
AND_SELL_RANGE = [5, 10, 15, 20, 25]
OR_SELL_RANGE = [30, 50, 70, 90, 120, 150, 200]
```

### 5. 自定义可视化

```bash
cd stock_strategy
python visualize_interval_buy_comparison.py NVDA s3 s5
python visualize_buy_strategy_comparison.py ALL
```

---

## 常见问题

### Q1: 为什么美股统一用 buy<0，而加密货币逐币设定？

**A**: 美股情绪分布相似 (均值接近 0)，统一阈值效果好。加密货币情绪分布差异大 (BTC 均值 11.9, SOL 均值 31.3)，需要逐币调整。

### Q2: 为什么 SOL 只用 2 年训练？

**A**: SOL 在 2021 年有百倍涨幅 ($1.5→$250)，包含这段数据会导致网格搜索选出过宽的 OR>200，影响后续窗口收益。用 2 年训练可以排除这个异常值。

### Q3: 训练模式为什么要与测试一致？

**A**: 如果训练用阈值分批、测试用间隔分批，两者买入时机不同，训练出的卖出参数可能不匹配。实验证明一致性训练能提升 +9.8% (加密货币)。

### Q4: 如何判断当前应该买入还是卖出？

**A**: 运行 `python3 retrain_params.py`，输出会显示每个资产的当前信号:
- `BUY MODE`: 情绪低于阈值，应买入
- `SELL (OR)`: 情绪超过 OR 阈值，应卖出
- `SELL (AND)`: 情绪超过 AND 且价格<MA50，应卖出
- `HOLD`: 无操作信号

### Q5: 季度更新会改变买入阈值吗？

**A**: 不会。买入阈值是固定的 (美股 buy<0, 加密货币逐币设定)。季度更新只更新卖出参数 (AND/OR 阈值)。

---

## 版本历史

- **v1.0** (2026-01-29): 从 AAPL 仓库分离
  - 美股情绪策略 (7只股票)
  - 加密货币情绪策略 (BTC/ETH/SOL)
  - 间隔分批买入 (v3.0)
  - 季度更新生产系统

---

**文档维护**: 2026-01-29
**来源**: 从 [AAPL](https://github.com/lslsbj89/AAPL) 仓库分离 (tag: v1.0-unified)
