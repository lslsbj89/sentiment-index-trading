# 快速索引 - Sentiment Strategy

**整理日期**: 2026-01-20

---

## 🚀 快速开始

### 我想知道结论 → 📖 看这里

```bash
# 1分钟速览（推荐！）
cat reports/SMOOTHING_COMPARISON_EXECUTIVE_SUMMARY.md

# 完整报告（详细版）
cat reports/SMOOTHING_FAIR_COMPARISON_FINAL_REPORT.md
```

**核心结论**:
- ✅ 个股优化 → 选 **Smoothing=3** (+16%)
- ✅ 统一参数 → 选 **Smoothing=5** (+61%)

---

### 我想看交易细节 → 💹 看这里

```bash
# 交易策略快速参考
cat reports/TRADE_ANALYSIS_QUICK_REFERENCE.md

# 详细交易分析
cat reports/TRADE_EXECUTION_ANALYSIS.md

# NVDA所有交易记录
cat results/detailed_trades_NVDA_s3_20260120_112432.csv
```

**关键数据**:
- NVDA: 15笔交易, 73.3%胜率, 424.5%收益 ⭐
- AAPL: 6笔交易, 83.3%胜率, 147.4%收益

---

### 我想看最优参数 → 🎯 看这里

```bash
# S3最优参数汇总
cat results/grid_search_s3_summary_20260120_110215.csv

# S5最优参数汇总
cat results/grid_search_s5_summary_20260120_110809.csv

# 公平对比结果
cat results/fair_comparison_optimized_20260120_110854.csv
```

**S3最优参数（个股优化）**:
```python
NVDA: buy<10, and>30, or>70
TSLA: buy<-10, and>25, or>50
AAPL: buy<-10, and>15, or>40
```

---

### 我想重新运行 → 🔧 看这里

```bash
# 公平对比S3 vs S5
python3 scripts/fair_comparison_s3_vs_s5.py

# 查看NVDA交易细节
python3 scripts/detailed_trade_analysis.py

# 多股票对比分析
python3 scripts/multi_symbol_trade_analysis.py

# S3参数搜索
python3 scripts/search_optimal_thresholds_s3.py

# S5参数搜索
python3 scripts/search_optimal_thresholds_s5.py
```

---

## 📂 文件夹导航

| 文件夹 | 内容 | 数量 |
|--------|------|------|
| **scripts/** | Python分析脚本 | 9个 |
| **results/** | CSV数据结果 | 32个 |
| **reports/** | Markdown报告 | 8个 |
| **charts/** | 可视化图表 | 1个 |
| **archives/** | 历史实验归档 | 2个 |
| **misc/** | 其他文件 | 1个 |

详细说明见: `DIRECTORY_STRUCTURE.md`

---

## ⭐ 重要文件清单

### 必读报告（按优先级）

1. `reports/SMOOTHING_COMPARISON_EXECUTIVE_SUMMARY.md` - 执行摘要 ⭐⭐⭐
2. `reports/TRADE_ANALYSIS_QUICK_REFERENCE.md` - 交易策略参考 ⭐⭐⭐
3. `reports/SMOOTHING_FAIR_COMPARISON_FINAL_REPORT.md` - 完整对比报告 ⭐⭐
4. `reports/TRADE_EXECUTION_ANALYSIS.md` - 详细交易分析 ⭐⭐

### 关键数据文件

1. `results/fair_comparison_optimized_20260120_110854.csv` - 个股优化对比
2. `results/fair_comparison_unified_20260120_110854.csv` - 统一参数对比
3. `results/grid_search_s3_summary_20260120_110215.csv` - S3最优参数
4. `results/grid_search_s5_summary_20260120_110809.csv` - S5最优参数
5. `results/detailed_trades_NVDA_s3_20260120_112432.csv` - NVDA交易记录

### 核心脚本

1. `scripts/fair_comparison_s3_vs_s5.py` - 公平对比（最重要）⭐
2. `scripts/detailed_trade_analysis.py` - 单股票分析
3. `scripts/multi_symbol_trade_analysis.py` - 多股票对比
4. `scripts/search_optimal_thresholds_s3.py` - S3参数搜索
5. `scripts/search_optimal_thresholds_s5.py` - S5参数搜索

---

## 📊 数据说明

### 参数网格搜索结果

**S3搜索范围**:
- buy: [-10, -5, 0, 5, 10]
- and: [15, 20, 25, 30]
- or: [40, 50, 60, 70]

**S5搜索范围**:
- buy: [-10, -5, 0, 5, 10]
- and: [15, 20, 25, 30]
- or: [35, 40, 45, 50]

**搜索结果**: 每个股票80个参数组合，共7只股票

### 公平对比场景

**场景1**: 个股优化 vs 个股优化
- S3: 每股用各自最优参数
- S5: 每股用各自最优参数
- 结果: S3 平均294.7%, S5 平均278.4% (+16.3%)

**场景2**: 统一参数 vs 统一参数
- S3: 所有股票用 buy<0, and>20, or>60
- S5: 所有股票用 buy<5, and>20, or>40
- 结果: S3 平均147.9%, S5 平均208.7% (-60.8%)

---

## 🔍 常见问题

### Q1: Smoothing=3 和 Smoothing=5 哪个好？

**A**: 取决于使用场景
- 如果能为每只股票单独优化参数 → **选S3** (+16%)
- 如果需要用统一参数管理多只股票 → **选S5** (+61%)

### Q2: 最优参数是什么？

**A**: 见 `results/grid_search_s3_summary_20260120_110215.csv`

**示例（NVDA）**:
- S3: buy<10, and>30, or>70 → 收益1016.5%
- S5: buy<10, and>30, or>50 → 收益1038.1%

### Q3: 如何查看具体交易？

**A**: 见 `results/detailed_trades_NVDA_s3_20260120_112432.csv`

包含每笔交易的：
- 买入时间、价格、指数、MA50
- 卖出时间、价格、指数、MA50、退出原因
- 持仓天数、收益率、收益金额

### Q4: 策略有什么风险？

**A**: 见 `reports/TRADE_EXECUTION_ANALYSIS.md`

**主要风险**:
1. 熊市脆弱（2022年NVDA 4笔中3笔亏损）
2. 缺乏止损（最大单笔亏损-38.96%）
3. 持仓固定（86.7%都是60天超时卖出）

**优化建议**:
- 加入-15%止损保护
- 根据指数动态调整持仓天数
- 分批建仓（越跌越买）

### Q5: 如何运行分析脚本？

**A**: 所有脚本在 `scripts/` 文件夹

```bash
cd scripts/
python3 fair_comparison_s3_vs_s5.py  # 运行公平对比
```

---

## 📞 联系与反馈

有问题或建议？参考以下文档：
- 完整文件夹结构: `DIRECTORY_STRUCTURE.md`
- 项目总览: `README.md`
- 实验日志: `reports/EXPERIMENT_LOG.md`

---

**提示**: 如果文件太多找不到，用这个命令搜索：
```bash
find . -name "*关键词*"
```

例如：
```bash
find . -name "*NVDA*"  # 查找所有包含NVDA的文件
find . -name "*summary*"  # 查找所有汇总文件
```
