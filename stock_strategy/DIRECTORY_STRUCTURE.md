# Sentiment Strategy 文件夹结构说明

**整理日期**: 2026-01-20
**目的**: 分类管理研究文件，便于查找和维护

---

## 📁 文件夹结构

```
sentiment_strategy/
├── README.md                          # 项目总览（根目录）
├── DIRECTORY_STRUCTURE.md             # 本文件（文件夹结构说明）
│
├── scripts/                           # Python脚本
│   ├── test_smoothing_*.py           # Smoothing参数测试脚本
│   ├── search_optimal_thresholds_*.py # 参数网格搜索脚本
│   ├── fair_comparison_s3_vs_s5.py   # S3 vs S5 公平对比脚本
│   ├── final_comparison_optimized_s3.py # S3优化后对比
│   ├── detailed_trade_analysis.py    # 单股票详细交易分析
│   └── multi_symbol_trade_analysis.py # 多股票对比分析
│
├── results/                           # 实验结果数据（CSV）
│   ├── grid_search_s3_*.csv          # S3网格搜索结果（各股详细+汇总）
│   ├── grid_search_s5_*.csv          # S5网格搜索结果（各股详细+汇总）
│   ├── fair_comparison_*.csv         # 公平对比结果
│   ├── mag7_*.csv                    # MAG7股票对比
│   ├── smoothing_comparison_*.csv    # Smoothing参数对比
│   ├── detailed_trades_*.csv         # 详细交易记录
│   └── multi_symbol_summary_*.csv    # 多股票汇总
│
├── reports/                           # 分析报告（Markdown）
│   ├── SMOOTHING_*.md                # Smoothing参数研究报告
│   │   ├── SMOOTHING_FAIR_COMPARISON_FINAL_REPORT.md  # 最终公平对比报告 ⭐
│   │   ├── SMOOTHING_COMPARISON_EXECUTIVE_SUMMARY.md  # 执行摘要
│   │   ├── SMOOTHING_FINAL_CONCLUSION.md              # 旧版结论（已过时）
│   │   ├── SMOOTHING_EXPERIMENT_RESULTS.md            # 初步实验结果
│   │   └── SMOOTHING_PARAMETER_ANALYSIS.md            # 参数理论分析
│   │
│   ├── TRADE_*.md                    # 交易执行分析报告
│   │   ├── TRADE_EXECUTION_ANALYSIS.md           # 详细交易分析 ⭐
│   │   └── TRADE_ANALYSIS_QUICK_REFERENCE.md     # 快速参考卡片
│   │
│   └── EXPERIMENT_LOG.md             # 实验日志（21个历史实验）
│
├── charts/                            # 可视化图表（PNG）
│   └── multi_symbol_comparison_*.png # 多股票对比图表
│
├── archives/                          # 历史实验归档
│   ├── exp_20260112_old_index/       # 实验：旧指数策略研究
│   └── exp_20260113_cycle_index/     # 实验：周期指数研究
│
└── misc/                              # 其他文件
    └── 10ma50.pine                   # TradingView Pine脚本
```

---

## 🗂️ 文件夹说明

### 📂 scripts/ - 脚本文件夹

**用途**: 存放所有Python分析脚本

| 脚本 | 功能 | 使用场景 |
|------|------|---------|
| `test_smoothing_*.py` | 测试不同Smoothing参数 | 初步对比S3 vs S5 |
| `search_optimal_thresholds_s3.py` | S3参数网格搜索 | 寻找S3最优阈值 |
| `search_optimal_thresholds_s5.py` | S5参数网格搜索 | 寻找S5最优阈值 |
| `fair_comparison_s3_vs_s5.py` | 公平对比脚本 ⭐ | 两个场景公平对比 |
| `detailed_trade_analysis.py` | 单股票详细分析 | 查看具体交易细节 |
| `multi_symbol_trade_analysis.py` | 多股票对比 | 生成对比图表 |

**运行示例**:
```bash
cd scripts/
python3 fair_comparison_s3_vs_s5.py  # 运行公平对比
python3 detailed_trade_analysis.py   # 查看NVDA交易细节
```

---

### 📂 results/ - 结果文件夹

**用途**: 存放所有CSV数据结果

**文件命名规则**:
- `grid_search_s3_{SYMBOL}_{timestamp}.csv` - S3网格搜索详细结果
- `grid_search_s3_summary_{timestamp}.csv` - S3网格搜索汇总
- `grid_search_s5_{SYMBOL}_{timestamp}.csv` - S5网格搜索详细结果
- `grid_search_s5_summary_{timestamp}.csv` - S5网格搜索汇总
- `fair_comparison_optimized_{timestamp}.csv` - 个股优化对比
- `fair_comparison_unified_{timestamp}.csv` - 统一参数对比
- `detailed_trades_{SYMBOL}_s3_{timestamp}.csv` - 详细交易记录

**关键文件**:
- `grid_search_s3_summary_20260120_110215.csv` - S3最优参数汇总
- `grid_search_s5_summary_20260120_110809.csv` - S5最优参数汇总
- `fair_comparison_optimized_20260120_110854.csv` - 最终公平对比（个股）
- `fair_comparison_unified_20260120_110854.csv` - 最终公平对比（统一）

**查看示例**:
```bash
cd results/
cat grid_search_s3_summary_20260120_110215.csv  # 查看S3最优参数
cat fair_comparison_optimized_20260120_110854.csv  # 查看公平对比结果
```

---

### 📂 reports/ - 报告文件夹

**用途**: 存放所有分析报告和结论文档

#### Smoothing参数研究系列

| 报告 | 状态 | 说明 |
|------|------|------|
| `SMOOTHING_FAIR_COMPARISON_FINAL_REPORT.md` | ✅ 最新 | 最终公平对比报告（推荐阅读）⭐ |
| `SMOOTHING_COMPARISON_EXECUTIVE_SUMMARY.md` | ✅ 最新 | 执行摘要（快速参考）⭐ |
| `SMOOTHING_FINAL_CONCLUSION.md` | ⚠️ 过时 | 旧版结论（不公平对比，已废弃） |
| `SMOOTHING_EXPERIMENT_RESULTS.md` | ⚠️ 过时 | 初步实验结果 |
| `SMOOTHING_PARAMETER_ANALYSIS.md` | ✅ 有效 | 参数理论分析 |

#### 交易执行分析系列

| 报告 | 说明 |
|------|------|
| `TRADE_EXECUTION_ANALYSIS.md` | 详细交易分析（推荐阅读）⭐ |
| `TRADE_ANALYSIS_QUICK_REFERENCE.md` | 快速参考卡片 |

#### 历史记录

| 报告 | 说明 |
|------|------|
| `EXPERIMENT_LOG.md` | 21个历史实验详细日志 |

**阅读建议**:
```bash
# 快速了解Smoothing结论
cat reports/SMOOTHING_COMPARISON_EXECUTIVE_SUMMARY.md

# 详细了解公平对比
cat reports/SMOOTHING_FAIR_COMPARISON_FINAL_REPORT.md

# 查看交易执行细节
cat reports/TRADE_EXECUTION_ANALYSIS.md

# 快速参考交易策略
cat reports/TRADE_ANALYSIS_QUICK_REFERENCE.md
```

---

### 📂 charts/ - 图表文件夹

**用途**: 存放所有可视化图表

| 图表 | 说明 |
|------|------|
| `multi_symbol_comparison_*.png` | 多股票对比图表（胜率、收益、交易次数等） |

---

### 📂 archives/ - 归档文件夹

**用途**: 存放历史实验完整目录

| 实验目录 | 说明 |
|---------|------|
| `exp_20260112_old_index/` | 旧指数策略研究（Experiment B） |
| `exp_20260113_cycle_index/` | 周期指数研究（Experiment C） |

**特点**: 每个实验目录包含完整的代码、数据、结果、图表

---

### 📂 misc/ - 其他文件夹

**用途**: 存放不属于上述分类的文件

| 文件 | 说明 |
|------|------|
| `10ma50.pine` | TradingView Pine脚本（10MA/50MA策略） |

---

## 🎯 快速查找指南

### 场景1: 我想知道Smoothing=3还是5更好？

```bash
# 方法1: 快速查看执行摘要
cat reports/SMOOTHING_COMPARISON_EXECUTIVE_SUMMARY.md

# 方法2: 详细了解公平对比
cat reports/SMOOTHING_FAIR_COMPARISON_FINAL_REPORT.md
```

**答案**:
- 个股优化 → 选 S3 (+16%)
- 统一参数 → 选 S5 (+61%)

---

### 场景2: 我想看NVDA的每笔交易细节

```bash
# 查看CSV数据
cat results/detailed_trades_NVDA_s3_20260120_112432.csv

# 查看分析报告
cat reports/TRADE_EXECUTION_ANALYSIS.md
```

---

### 场景3: 我想知道各股票的最优参数

```bash
# S3最优参数
cat results/grid_search_s3_summary_20260120_110215.csv

# S5最优参数
cat results/grid_search_s5_summary_20260120_110809.csv
```

---

### 场景4: 我想重新运行分析

```bash
# 运行公平对比
python3 scripts/fair_comparison_s3_vs_s5.py

# 查看NVDA交易细节
python3 scripts/detailed_trade_analysis.py

# 多股票对比分析
python3 scripts/multi_symbol_trade_analysis.py
```

---

### 场景5: 我想查看参数搜索结果

```bash
# 查看NVDA的S3网格搜索详细结果
cat results/grid_search_s3_NVDA_20260120_110215.csv

# 查看所有股票的S3最优参数汇总
cat results/grid_search_s3_summary_20260120_110215.csv
```

---

## 📊 推荐阅读顺序

### 初次了解项目
1. `README.md` (根目录) - 项目总览
2. `reports/SMOOTHING_COMPARISON_EXECUTIVE_SUMMARY.md` - 核心结论
3. `reports/TRADE_ANALYSIS_QUICK_REFERENCE.md` - 交易策略快速参考

### 深入研究
1. `reports/SMOOTHING_FAIR_COMPARISON_FINAL_REPORT.md` - 完整对比报告
2. `reports/TRADE_EXECUTION_ANALYSIS.md` - 详细交易分析
3. `reports/EXPERIMENT_LOG.md` - 历史实验记录

### 数据分析
1. `results/fair_comparison_optimized_20260120_110854.csv` - 公平对比数据
2. `results/grid_search_s3_summary_20260120_110215.csv` - S3最优参数
3. `results/detailed_trades_NVDA_s3_20260120_112432.csv` - 交易明细

---

## 🔧 维护说明

### 新增实验时

1. **脚本**: 放入 `scripts/`
2. **结果CSV**: 放入 `results/`
3. **报告MD**: 放入 `reports/`
4. **图表PNG**: 放入 `charts/`
5. **实验目录**: 放入 `archives/`

### 文件命名规范

- **脚本**: `{功能}_{版本}.py` (如 `test_smoothing_final.py`)
- **CSV**: `{类型}_{内容}_{timestamp}.csv` (如 `grid_search_s3_NVDA_20260120_110215.csv`)
- **报告**: `{主题}_{类型}.md` (如 `SMOOTHING_COMPARISON_EXECUTIVE_SUMMARY.md`)
- **图表**: `{内容}_{timestamp}.png` (如 `multi_symbol_comparison_20260120_112515.png`)

### 定期清理

- **重复文件**: 保留最新timestamp版本
- **过时报告**: 标注状态（✅最新/⚠️过时/❌废弃）
- **临时文件**: 删除或移至 `misc/`

---

## 📝 版本历史

| 日期 | 变更 |
|------|------|
| 2026-01-20 | 初次整理，创建分类文件夹结构 |

---

**整理完成！** 现在文件夹结构清晰，易于查找和维护。
