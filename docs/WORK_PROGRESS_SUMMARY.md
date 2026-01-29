# 工作进度总结

**更新日期**: 2026-01-28
**项目**: 情绪策略研究 (美股七强 + 加密货币)
**数据源**: Fear Greed Index S3 (美股) / yahoo_artemis_index (加密货币)

---

## 一、策略演进路线

```
v1.0 原始策略        → 单次买入 + OR/AND卖出
v2.0 阈值分批策略    → sent<5/0/-5/-10, 每档25% (4批)
v3.0 间隔买入策略    → sent<0进入, 每7天买20%, 回升全买 ⭐当前生产版本
```

---

## 二、实验记录

### 2.1 买入策略对比 (v2.0基础)

对比了5种买入策略，使用Walk-Forward验证 (4年训练 + 1年测试):

| 策略 | 说明 | 效果 |
|------|------|------|
| A. 基准策略 | 单次买入80% | 基准 |
| B. MA确认策略 | 需价格>MA10确认 | 一般 |
| C. 时间分批策略 | 首次40%，5天后40% | 一般 |
| D. 阈值分批策略 | sent<5/0/-5/-10，每档25% | 优 |
| **E. 间隔买入策略** | sent<0→20%/7天, 回升全买 | **最优** |

### 2.2 MF26 vs MF13 对比 (`exp_fear_greed_mf26/`)

对比 MoneyFlow=26 与 MoneyFlow=13 两组参数:

| 指标 | MF=13 | MF=26 |
|------|-------|-------|
| 平均收益 | +303.2% | +291.7% |
| 稳定性 | 更高 | 较低 |
| TSLA差异 | +245.9% | +217.9% |

**结论**: 选择 MF=13 用于生产 (更稳定, 7只股票中5只更优)

### 2.3 MA50斜率增强 (`exp_and_ma50_slope/`)

在AND卖出条件上增加MA50斜率<0的要求:

**结论**: 改善不显著, 增加复杂度, 不采用

### 2.4 间隔买入策略 (`exp_interval_buy/`) ⭐

v3.0核心改进 — 解决v2.0的两个问题:
- 问题1: 4批阈值(5/0/-5/-10)在温和下跌中只触发1-2批, 资金利用率低
- 问题2: 阈值在训练/实际中匹配度不稳定

**v3.0机制**:
```
情绪 < 0 → 进入买入模式, 买20%
每隔7天 → 再买20% (最多5批=100%)
情绪 >= 0 → 买入全部剩余, 退出
```

**Walk-Forward回测结果 (7只股票, 7个窗口)**:

| 股票 | 平均收益 | 最差窗口 | 最优窗口 |
|------|---------|---------|---------|
| NVDA | +432.4% | +323.0% | +573.4% |
| TSLA | +251.2% | +133.9% | +549.2% |
| AAPL | +209.6% | +118.7% | +331.4% |
| MSFT | +140.9% | +59.7% | +263.8% |
| GOOGL | +105.3% | +58.8% | +179.5% |
| AMZN | +97.9% | +10.1% | +276.9% |
| META | +94.0% | -1.8% | +128.0% |

### 2.5 买入阈值对比

测试5个买入阈值 [-15, -10, -5, 0, 5], 7只股票×7窗口:

| 阈值 | 平均收益 | 排名 |
|------|---------|------|
| buy<-15 | +277.7% | 5 |
| buy<-10 | +339.8% | 4 |
| buy<-5 | +400.0% | 3 |
| **buy<0** | **+431.5%** | **1** |
| buy<5 | +410.9% | 2 |

**结论**: buy<0 为最优统一阈值

### 2.6 卖出参数网格搜索

AND×OR 25组合网格搜索, W2026训练期 (2022-2025), **interval buy training**:

| 股票 | AND | OR | 训练收益 | 特征 |
|------|-----|-----|---------|------|
| NVDA | >5 | >50 | +517.9% | 敏感, 快进快出 |
| TSLA | >25 | >30 | +233.6% | AND宽松, OR敏感 |
| AAPL | >5 | >30 | +111.4% | 最敏感 |
| MSFT | >5 | >30 | +75.0% | 敏感, 快进快出 |
| GOOGL | >10 | >50 | +137.4% | 中等偏松 |
| AMZN | >10 | >30 | +88.3% | 中等 |
| META | >15 | >55 | +141.9% | 稳定 |

> 已从 staged buy training 改为 interval buy training。3只股票参数变化: TSLA OR 60→30, MSFT AND 15→5, GOOGL AND 15→10。

### 2.7 情绪指数四方对比 (`exp_sentiment_index_comparison/`) ⭐

用间隔分批策略，7只股票全量对比 S3/S5/S3+VIX/S5+VIX 四种情绪指数:

| 对比组 | 胜负 | 结论 |
|--------|------|------|
| S3 vs S3+VIX | S3 4:3 S3+VIX | VIX对高beta有帮助, 对低beta有害 |
| S5 vs S5+VIX | S5 4:3 S5+VIX | 同上 |
| S3 vs S5 | S3 5:2 S5 | S3更稳定 |

**总结**: S3 最均衡 (avg +431.5%), S5+VIX 均值最高 (+570.6%) 但被 NVDA 拉高, 不稳定

**结论: S3 确认为生产方案**

### 2.8 阈值分批 vs 间隔分批对比 (`exp_buy_strategy_staged_vs_interval/`) ⭐

直接对比 v2.0 阈值分批与 v3.0 间隔分批的效果:

| 策略 | 合计资产 | 平均收益 | 胜场 |
|------|---------|---------|------|
| v2.0 阈值分批 | $2,966,082 | +323.7% | 1 (TSLA) |
| **v3.0 间隔分批** | **$3,720,269** | **+431.5%** | **6** |

间隔分批多赚 $754,187 (+25.4%), 6:1 全面胜出。

TSLA 是唯一 v2.0 胜出的股票 (+492.9% vs +413.3%), 原因是 W2020 冷启动期 AND>15 过早卖出后 TSLA 暴涨 +543%, walk-forward 在后续窗口已自动修正。

**结论: v3.0 间隔分批确认为生产方案, 维持现状不做修改**

### 2.9 通用可视化工具

创建了两个通用情绪指数对比可视化脚本:

| 脚本 | 买入策略 | 用途 |
|------|---------|------|
| `visualize_sentiment_comparison.py` | v2.0 阈值分批 | 阈值分批版对比 |
| `visualize_interval_buy_comparison.py` | v3.0 间隔分批 | 间隔分批版对比 |
| `visualize_buy_strategy_comparison.py` | 两者对比 | 买入策略对比 |

支持5种情绪指数: s3, s5, mf26, s3_vix, s5_vix

用法: `python visualize_sentiment_comparison.py NVDA s3 s5` 或 `... ALL s3 s5`

### 2.10 加密货币情绪策略 (`crypto_strategy/`) ⭐

将美股情绪策略移植到加密货币 (BTC, ETH, SOL)，使用 yahoo_candles (价格) + yahoo_artemis_index (情绪, smoothing=3)。

**数据特征**:

| 币种 | 情绪均值 | 情绪范围 | 数据起始 |
|------|---------|---------|---------|
| BTC | 11.9 | -37~+173 | 2014 |
| ETH | 8.1 | -47~+130 | 2018 |
| SOL | 31.3 | -50~+392 | 2020 |

**Walk-Forward设计**: 测试期统一 2023-2025, BTC/ETH 3年训练, SOL 2年训练

**最终结果 (间隔分批 3:0 胜出)**:

| 币种 | v2.0 阈值分批 | v3.0 间隔分批 | 最优阈值 |
|------|-------------|-------------|---------|
| BTC | +50.1% ($150K) | **+344.3% ($444K)** | buy<10 |
| ETH | +60.8% ($161K) | **+90.3% ($190K)** | buy<-10 |
| SOL | +1000.3% ($1.10M) | **+1195.4% ($1.30M)** | buy<-10 |
| **合计** | **$1,411K** | **$1,930K** | |

**与美股的关键差异**:
- 买入阈值需要逐币设定 (美股统一 buy<0)
- OR卖出范围需大幅扩展: SOL 用到 OR>200 (美股最高 OR>60)
- BTC 情绪偏高, 阈值分批 [5,0,-5,-10] 几乎无法触发买入

### 2.11 训练模式修正实验 (`crypto_strategy/`) ⭐

发现 v1 代码的一个不合理之处: 搜索间隔分批的卖出参数时, 训练阶段用的是阈值分批买入 [5,0,-5,-10], 与实际测试策略不匹配。

对比了3种训练模式:

| 训练模式 | 说明 | 合计资产 | vs 旧版 |
|---------|------|---------|---------|
| Staged (旧) | 训练用阈值分批 | $1,757,301 | 基准 |
| **Interval (新)** | **训练用间隔分批 (与测试一致)** | **$1,929,918** | **+9.8%** |
| Oneshot | 训练用一次性买卖 | $1,751,025 | -0.4% |

**关键改善**: SOL W2023 训练出 AND>5 (旧版 AND>15), 持仓更久, +848.5% vs +691.5%

**结论**: 训练策略必须与测试策略一致, 已更新为 Interval Train

### 2.12 加密货币训练周期对比

测试3种训练窗口配置对 SOL 的影响:

| 配置 | BTC | ETH | SOL | 合计 |
|------|-----|-----|-----|------|
| 2年全部 | +280.3% ($380K) | +111.9% ($212K) | +1195.4% ($1,295K) | $1,888K |
| **3年BTC/ETH + 2年SOL** | **+344.3% ($444K)** | **+90.3% ($190K)** | **+1195.4% ($1,295K)** | **$1,930K** |
| 3年全部 | +344.3% ($444K) | +90.3% ($190K) | +889.2% ($989K) | $1,624K |

**关键发现**: SOL 用3年训练反而亏损 $306K。原因: 3年窗口包含 2021 年 SOL 百倍涨幅 ($1.5→$250), 导致网格搜索选出过宽的 OR>200, W2024 收益从 +43.9% 降至 +9.9%。

**结论**: BTC/ETH 用 3 年训练, SOL 用 2 年训练 (排除 2021 异常值)

### 2.13 加密货币生产部署 (`crypto_strategy/2026_production/`) ⭐

建立了加密货币季度更新生产系统, 与美股生产系统同构:

**Q1-2026 卖出参数** (interval buy training):

| 币种 | Buy< | AND | OR | 训练收益 | 训练期 |
|------|------|-----|-----|---------|--------|
| BTC | 10 | >20 | >50 | +352.3% | 2023-2025 (3yr) |
| ETH | -10 | >5 | >30 | +148.2% | 2023-2025 (3yr) |
| SOL | -10 | >15 | >30 | +104.4% | 2024-2025 (2yr) |

**当前信号** (2026-01-24): 3 币种全部处于 **BUY MODE**

### 2.14 美股训练模式对比 (`sentiment_strategy/interval_train_experiment/`) ⭐

将 crypto 的训练模式修正方案应用到美股, 对比阈值分批训练 vs 间隔分批训练:

| 股票 | Staged Train | Interval Train | Diff | Winner |
|------|-------------|---------------|------|--------|
| NVDA | +1510.7% | +1510.7% | 0.0% | Tie |
| TSLA | +413.3% | +413.3% | 0.0% | Tie |
| AAPL | +275.6% | +310.0% | +34.4% | Interval |
| MSFT | +219.8% | +225.0% | +5.2% | Interval |
| GOOGL | +269.0% | +280.3% | +11.4% | Interval |
| AMZN | +171.4% | +215.1% | +43.7% | Interval |
| META | +160.5% | +122.0% | -38.5% | Staged |
| **合计** | **$3,720K** | **$3,776K** | **+$56K** | **Interval 4:1** |

**改进幅度 +1.5%**, 远小于 crypto 的 +9.8%。原因:
- 美股 OR 范围窄 (30-60), 两种训练模式选出的参数经常一致
- Crypto OR 范围宽 (30-200), 训练模式差异被放大

---

## 三、2026生产方案

### 3.1 资产配置

```
NVDA 25% + TSLA 15% + AAPL 15% + GOOGL 15% + MSFT 10% + AMZN 10% + 现金 10%
```

### 3.2 买入策略 (固定, 不需要训练)

| 参数 | 值 |
|------|------|
| 买入触发 | 情绪 < 0 |
| 间隔天数 | 7 天 |
| 每批比例 | 20% |

### 3.3 卖出策略 (每季度更新)

双条件卖出: OR (情绪>阈值→无条件卖) + AND (情绪>阈值 且 价格<MA50→卖)

当前参数 (2026-Q1, 训练期2022-2025, interval buy training) 见上方2.6节

### 3.4 季度更新

```bash
cd 2026_stock_production
python3 retrain_sell_params.py          # 自动用最近4年
python3 retrain_sell_params.py 2022 2026  # 或指定训练期
```

| 时间 | 训练期 | 操作 |
|------|--------|------|
| 2026-01 (已完成) | 2022-2025 | 当前参数 |
| 2026-04 | 2022-2026 Q1 | 重新训练 |
| 2026-07 | 2022-2026 H1 | 重新训练 |
| 2026-10 | 2023-2026 Q3 | 重新训练 |

---

## 四、产出文件清单

### 4.1 美股生产文件夹 (`2026_stock_production/`) ⭐

| 文件 | 说明 |
|------|------|
| `OPERATION_GUIDE.md` | 简洁版每日操作指南 |
| `2026_INTERVAL_BUY_RECOMMENDATION.md` | 完整版策略推荐文档 |
| `retrain_params.py` | 季度参数更新脚本 |
| `params_*.csv` | 买入+卖出参数文件 |

### 4.2 实验文件夹 (`sentiment_strategy/`)

| 目录 | 说明 |
|------|------|
| `exp_buy_strategy_comparison/` | v2.0 买入策略对比 |
| `exp_fear_greed_mf26/` | MF26 vs MF13 对比 |
| `exp_and_ma50_slope/` | MA50斜率增强实验 |
| `exp_interval_buy/` | v3.0 间隔买入策略 |
| `exp_sentiment_index_comparison/` | S3/S5/S3+VIX/S5+VIX 四方对比 |
| `exp_buy_strategy_staged_vs_interval/` | 阈值分批 vs 间隔分批对比 |

### 4.3 加密货币策略 (`crypto_strategy/`) ⭐

| 文件 | 说明 |
|------|------|
| `crypto_backtest.py` | 主回测脚本 (阈值分批 vs 间隔分批, interval训练) |
| `crypto_threshold_search.py` | 买入阈值搜索 (7个候选值) |
| `crypto_train_mode_comparison.py` | 训练模式对比 (staged/interval/oneshot) |
| `crypto_backtest_v1.py` | 备份: staged训练版本 |
| `crypto_threshold_search_v1.py` | 备份: staged训练版本 |
| `CRYPTO_BACKTEST_REPORT.md` | 完整回测报告 |
| `*_staged_vs_interval.png` (x3) | 逐币对比图 |
| `crypto_threshold_search.png` | 阈值搜索对比图 |
| `crypto_train_mode_comparison.png` | 训练模式对比图 |

### 4.4 加密货币生产文件夹 (`2026_crypto_production/`) ⭐

| 文件 | 说明 |
|------|------|
| `OPERATION_GUIDE.md` | 加密货币操作指南 |
| `retrain_params.py` | 季度参数更新脚本 (interval buy training) |
| `params_*.csv` | 买入+卖出参数文件 |

### 4.5 美股训练模式实验 (`sentiment_strategy/interval_train_experiment/`)

| 文件 | 说明 |
|------|------|
| `train_mode_comparison.py` | Staged vs Interval 训练模式对比 |
| `train_mode_comparison_s3.png` | 收益对比图 |
| `param_diff_s3.png` | 参数差异矩阵图 |

### 4.6 可视化工具

| 文件 | 说明 |
|------|------|
| `visualize_sentiment_comparison.py` | 情绪指数对比 (阈值分批版) |
| `visualize_interval_buy_comparison.py` | 情绪指数对比 (间隔分批版) |
| `visualize_buy_strategy_comparison.py` | 买入策略对比 (阈值 vs 间隔) |

### 4.7 根目录文档

| 文件 | 说明 |
|------|------|
| `2026_THRESHOLD_STAGED_RECOMMENDATION.md` | v2.0推荐 (已被v3.0取代) |
| `WORK_PROGRESS_SUMMARY.md` | 本文档 |

---

## 五、Git提交历史

```
163fbae docs: add project overview to CLAUDE.md
ab37f33 fix: allow fractional crypto purchases in all crypto strategy files
2ddadd0 fix: allow fractional crypto purchases in retrain_params.py
dddab86 refactor: rename params files and add buy thresholds
efbbc01 docs: update work progress with production folder relocation
b7ccd9c refactor: move production folders to root and add per-stock buy thresholds
538bcad docs: update work progress with US stock interval buy training
a704a29 feat: switch US stock production to interval buy training
7edbfb4 docs: update work progress with production deploy and train mode experiments
e084c8f feat: add US stock train mode comparison experiment
a16de73 feat: add crypto production system with quarterly retraining
838e1e0 docs: update work progress with crypto experiments and training mode fix
577fbf1 feat: fix training mode mismatch — use interval buy for interval test
0fc823a feat: revise crypto backtest with unified test period 2023-2025
ea245e2 feat: add crypto sentiment strategy backtest (BTC/ETH/SOL)
082fc6a docs: update work progress summary with today's experiments
3038efe feat: add staged vs interval buy strategy comparison (7 stocks)
2b9258c rename: exp_s3_vs_s3_vix -> exp_sentiment_index_comparison
965958e feat: expand to four-way comparison (S3/S5/S3+VIX/S5+VIX)
0fd7976 feat: add S3 vs S3+VIX comparison experiment (7 stocks, interval buy)
8eadb18 feat: add interval buy version of sentiment comparison visualization
fd46190 feat: add generic sentiment index comparison visualization
59a0965 docs: add work progress summary
8daa8a5 feat: add 2026 production folder with operation guide and retrain script
0a98962 feat: add W2026 sell parameters from grid search (train 2022-2025)
14b64c7 docs: integrate buy threshold comparison into 2026 recommendation
c20ff0c docs: add 2026 interval buy strategy recommendation
1a06fa9 docs: move conclusions to top of interval buy summary
1fdfd7b docs: add buy threshold comparison and interval buy strategy summary
6772444 feat: add joint optimization experiment for interval buy (NVDA)
55aed86 feat: run interval buy strategy on all 7 stocks
a150397 feat: add interval buy strategy experiment (NVDA test)
94464c4 feat: add MA50 slope enhanced AND sell condition experiment
88dfcb0 docs: finalize MF26 experiment - select MF=13 for production
811b5d9 docs: add AAPL MF13 vs MF26 trading visualization
28224c9 feat: add MF13 vs MF26 trading process visualization for TSLA
cdc4ec4 docs: add TSLA root cause analysis to MF26 comparison report
5788ad1 feat: optimize MF=26 sell threshold with AND>=15 constraint
2ba6564 feat: add MoneyFlow=26 experiment and MF26 vs MF13 comparison report
```

---

## 六、关键结论

### 美股
1. **情绪指数**: S3 (Smoothing=3, MoneyFlow=13) 最均衡稳定, 确认为生产方案
2. **买入策略**: v3.0 间隔分批 (sent<0, 每7天20%, 回升全买) 以 6:1 胜出, 确认为生产方案
3. **卖出策略**: OR/AND 双条件, 训练期网格搜索, 每季度更新
4. **VIX因子**: 对高beta股票有帮助, 对低beta有害, 不纳入生产方案
5. **TSLA异常**: W2020 过早卖出是冷启动期个案, walk-forward 已自动修正, 无需人工干预

### 加密货币
6. **间隔分批 3:0 胜出**: 合计 $1.93M vs $1.41M (阈值分批), SOL 贡献最大 (+1195%)
7. **买入阈值需逐币设定**: BTC buy<10, ETH buy<-10, SOL buy<-10 (美股统一 buy<0)
8. **OR卖出范围扩展**: SOL 需要 OR>150~200 (美股仅 OR>30~60)
9. **训练模式须匹配测试**: 间隔分批训练比阈值分批训练多赚 9.8% ($172K, crypto) / +1.5% ($56K, 美股)
10. **训练周期逐资产选择**: SOL 用 2 年 (排除 2021 百倍异常), BTC/ETH 用 3 年 (覆盖更多周期)

## 七、后续工作

### 美股
1. **每日执行**: 按 `2026_stock_production/OPERATION_GUIDE.md` 检查情绪信号
2. **季度更新**: 在 `2026_stock_production/` 运行 `retrain_sell_params.py` 更新卖出参数
3. **市场适应**: 牛市考虑提高OR阈值, 熊市增加现金储备
4. **策略迭代**: 积累2026实盘数据后评估策略效果

### 加密货币
5. **生产部署**: ✅ 已建立 `2026_crypto_production/` (已移至主目录), 含操作指南和季度更新脚本
6. **季度更新**: 2026-04 在 `2026_crypto_production/` 运行 `retrain_sell_params.py` 更新卖出参数 (含 Q1 新数据)
7. **扩展币种**: 考虑增加更多主流币种 (如 DOGE, AVAX 等)

### 训练模式
8. **美股生产方案更新**: ✅ 已将美股 `retrain_sell_params.py` 改用 interval buy training, 3只股票参数变化 (TSLA/MSFT/GOOGL)
9. **进一步实验**: 对美股测试更多情绪指数 (s5, s3_vix) 下的训练模式对比

---

*报告生成: Claude Code*
*最后更新: 2026-01-29*
