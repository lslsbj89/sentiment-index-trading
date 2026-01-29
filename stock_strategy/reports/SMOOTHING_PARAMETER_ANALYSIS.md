# Smoothing 参数分析报告

**日期**: 2026-01-20
**状态**: 待测试
**目的**: 分析不同 smoothing 参数对情绪指数策略的影响

---

## 背景

### 当前使用的 Smoothing 值

所有现有实验都使用 **smoothing = 5** (RMA平滑周期)

| 指数版本 | Smoothing值 | 数据量 | 使用情况 |
|---------|------------|--------|---------|
| fear_greed_index | 5 | 132,543 条 | ✅ 当前使用 |
| fear_greed_index_backup_20260113 | 5 | - | ✅ 当前使用 |

### TradingView Pine 脚本

在 `10ma50.pine` 中，smoothing 参数是可配置的：
```pine
smoothing = input(5, title="Smoothing", options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
```

但所有Python回测实验都固定使用 **smoothing=5**，没有测试其他值。

---

## 为什么要测试 Smoothing = 3？

### 1. RMA 计算原理

RMA (Wilder's Moving Average) 计算公式：
```
RMA[i] = alpha * value[i] + (1 - alpha) * RMA[i-1]
其中 alpha = 1 / period
```

**不同 smoothing 值的 alpha**:
- smoothing=3: alpha = 0.333 (33.3%新值权重)
- smoothing=5: alpha = 0.200 (20.0%新值权重) ← **当前**
- smoothing=7: alpha = 0.143 (14.3%新值权重)
- smoothing=10: alpha = 0.100 (10.0%新值权重)

### 2. Smoothing 值的影响

| 特性 | smoothing=3 | smoothing=5 | smoothing=7 | smoothing=10 |
|------|------------|------------|------------|-------------|
| **响应速度** | ⚡ 非常快 | 🔥 快 | 🐢 慢 | 🐌 很慢 |
| **信号灵敏度** | 📈 极高 | 📊 高 | 📉 中 | 📉 低 |
| **假信号频率** | ⚠️ 较多 | ✅ 适中 | ✅ 较少 | ✅ 很少 |
| **趋势滞后性** | ✅ 极小 | ✅ 小 | ⚠️ 中 | ⚠️ 大 |
| **交易频率** | 📈 很高 | 📊 高 | 📉 中 | 📉 低 |
| **平滑程度** | ⚠️ 抖动多 | ✅ 平滑 | ✅ 很平滑 | ✅ 极平滑 |

### 3. 理论预期

#### Smoothing = 3 的优势

1. **更快捕捉市场情绪变化**
   - 市场恐慌时，能更早发出买入信号
   - 市场贪婪时，能更早发出卖出信号

2. **减少趋势滞后**
   - 当前smoothing=5可能错过最佳入场/出场点
   - smoothing=3能更及时响应

3. **适合高波动股票**
   - TSLA、NVDA等高波动股可能受益
   - 快速进出，捕捉波段

#### Smoothing = 3 的劣势

1. **假信号增多**
   - 短期情绪波动可能触发不必要的交易
   - 增加交易成本（手续费）

2. **频繁交易**
   - 可能导致过度交易
   - 心理压力增大

3. **噪音干扰**
   - 可能被短期噪音误导
   - 影响长期趋势判断

---

## 预期测试结果

### 假设 1: 高波动股 (TSLA, NVDA) 受益

**理由**:
- 高波动股情绪变化快
- smoothing=3能更快捕捉转折点
- 减少"错过最佳入场点"的情况

**预期**:
- TSLA: 收益提升 10-20%
- NVDA: 收益提升 5-15%
- 交易次数增加 50-100%

### 假设 2: 低波动股 (GOOGL, AAPL) 受损

**理由**:
- 低波动股情绪变化慢
- smoothing=3会产生更多假信号
- 增加不必要的交易成本

**预期**:
- GOOGL: 收益下降 5-10%
- AAPL: 收益下降 3-8%
- 交易次数增加 100-200%

### 假设 3: 整体夏普率下降

**理由**:
- 交易频率增加 → 手续费增加
- 假信号增多 → 胜率下降
- 回撤波动增大

**预期**:
- 平均夏普率: 0.74 → 0.65~0.70
- 平均胜率: 89% → 75~85%

---

## 测试方案

### 测试维度

| 维度 | 指标 | 对比方式 |
|------|------|---------|
| **收益率** | total_return | smoothing=3 vs 5 |
| **风险调整收益** | sharpe_ratio | smoothing=3 vs 5 |
| **最大回撤** | max_drawdown | smoothing=3 vs 5 |
| **交易频率** | num_trades | smoothing=3 vs 5 |
| **胜率** | win_rate | smoothing=3 vs 5 |
| **平均持仓天数** | avg_holding_days | smoothing=3 vs 5 |

### 测试股票

**MAG7 全量测试**:
- NVDA (高波动)
- TSLA (高波动)
- META (中波动)
- GOOGL (低波动)
- AAPL (低波动)
- MSFT (低波动)
- AMZN (中波动)

### 测试周期

- **主测试**: 2021-2025 (5年)
- **样本外**: 2016-2020 (5年)

### Smoothing 值测试

| 测试组 | Smoothing值 | 目的 |
|--------|------------|------|
| 组1 | 3 | 高灵敏度 |
| 组2 | 5 | 当前基准 ✅ |
| 组3 | 7 | 低灵敏度 |
| 组4 | 10 | 极低灵敏度 |

---

## 运行测试脚本

### 快速开始

```bash
cd /Users/sc2025/Desktop/test/AAPL/sentiment_strategy
python test_smoothing_parameter.py
```

### 实验内容

**实验1**: TSLA Smoothing 参数对比 (3, 5, 7, 10)
- 输出: `smoothing_comparison_TSLA_*.csv`

**实验2**: MAG7 Smoothing=3 测试
- 输出: `mag7_smoothing3_*.csv`

**实验3**: MAG7 Smoothing=5 测试（对照组）
- 输出: `mag7_smoothing5_*.csv`

**实验4**: Smoothing=3 vs 5 对比分析
- 输出: `smoothing_comparison_mag7_*.csv`

### 预计运行时间

- 实验1: ~5 分钟
- 实验2: ~15 分钟
- 实验3: ~15 分钟
- **总计**: ~35 分钟

---

## 数据生成方案

### 选项1: 直接从raw_index计算 ✅ 推荐

**优势**:
- 不需要修改数据库
- 灵活测试多个smoothing值
- 快速迭代

**实现**:
```python
def generate_smoothed_index(raw_df, smoothing=5):
    """根据raw_index生成不同smoothing的平滑指数"""
    alpha = 1.0 / smoothing
    smoothed = raw_df['raw_index'].ewm(alpha=alpha, adjust=False).mean()
    return smoothed
```

### 选项2: 生成新的数据库表

**优势**:
- 一次计算，多次使用
- 与现有代码一致

**劣势**:
- 需要修改数据库
- 磁盘空间占用

**实现**:
```sql
CREATE TABLE fear_greed_index_smoothing3 AS
SELECT
    *,
    -- 重新计算smoothed_index
FROM fear_greed_index;
```

---

## 预期发现

### 可能的结论 A: Smoothing=3 整体更优

**如果结果显示**:
- 平均收益: +230% (vs +208% @ smoothing=5)
- 平均夏普: 0.80 (vs 0.74 @ smoothing=5)
- 交易次数: +50%，但胜率保持

**结论**:
- ✅ 建议全面切换至 smoothing=3
- ✅ 更新所有策略配置
- ✅ 重新生成数据库索引

### 可能的结论 B: Smoothing=3 仅适合高波动股

**如果结果显示**:
- NVDA/TSLA: 收益提升 15%+
- GOOGL/AAPL: 收益下降 5%+
- 整体: 收益持平，夏普下降

**结论**:
- ⚠️ 建议分股票类型使用不同smoothing
- 高波动股: smoothing=3
- 低波动股: smoothing=5

### 可能的结论 C: Smoothing=5 仍然最优

**如果结果显示**:
- 平均收益: +195% (vs +208% @ smoothing=5)
- 平均夏普: 0.68 (vs 0.74 @ smoothing=5)
- 交易次数: +100%，胜率下降

**结论**:
- ✅ 维持 smoothing=5
- ❌ smoothing=3 假信号太多
- 📝 记录实验结果，不采纳

---

## 后续研究方向

### 1. 动态 Smoothing

根据市场波动率动态调整:
```python
def adaptive_smoothing(volatility):
    if volatility > 0.5:
        return 3  # 高波动用3
    elif volatility > 0.3:
        return 5  # 中波动用5
    else:
        return 7  # 低波动用7
```

### 2. 双指数融合

同时使用两个smoothing值:
```python
fast_signal = smoothed_index(raw, smoothing=3)  # 快信号
slow_signal = smoothed_index(raw, smoothing=7)  # 慢信号

# 金叉买入，死叉卖出
buy = (fast_signal crosses above slow_signal) and (fast_signal < threshold)
sell = (fast_signal crosses below slow_signal) or (fast_signal > threshold)
```

### 3. 个股化 Smoothing

为每只股票优化专属smoothing值:
```python
optimal_smoothing = {
    'NVDA': 3,    # 高波动
    'TSLA': 3,    # 高波动
    'META': 4,    # 中波动
    'GOOGL': 6,   # 低波动
    'AAPL': 5,    # 中低波动
    'MSFT': 5,    # 中低波动
    'AMZN': 4     # 中波动
}
```

---

## 关键问题

### Q1: 为什么之前选择 smoothing=5？

**A**: 可能原因：
1. TradingView默认值
2. 经验值（常用EMA/RMA周期）
3. 未进行系统性测试

### Q2: Smoothing 会影响阈值吗？

**A**: 会！
- smoothing=3 → 指数波动大 → 可能需要调整阈值
- 例如：原来 buy<5, sell>20，可能需要 buy<3, sell>18

### Q3: 需要重新优化参数吗？

**A**: 建议重新搜索：
- 买入阈值: -5 ~ 10
- AND卖出阈值: 10 ~ 25
- OR卖出阈值: 30 ~ 45

---

## 实验检查清单

- [ ] 运行 test_smoothing_parameter.py
- [ ] 收集所有CSV结果文件
- [ ] 分析各股票表现差异
- [ ] 对比 smoothing=3 vs 5 整体效果
- [ ] 检查交易频率变化
- [ ] 检查胜率变化
- [ ] 检查夏普率变化
- [ ] 决定是否采纳 smoothing=3
- [ ] 如果采纳，重新优化阈值参数
- [ ] 更新策略文档

---

## 参考资料

### RMA vs EMA

| 指标 | 公式 | Alpha | 特点 |
|------|------|-------|------|
| **EMA** | α=2/(n+1) | 0.333 (n=5) | 标准指数移动平均 |
| **RMA** | α=1/n | 0.200 (n=5) | Wilder平滑，更平滑 |

### Smoothing 学术研究

- **短周期 (3-5)**: 适合短线交易，捕捉快速变化
- **中周期 (5-10)**: 适合波段交易，平衡灵敏度和稳定性
- **长周期 (10+)**: 适合长线投资，过滤噪音

### 现有策略参数

```python
# 新版指数 (smoothing=5)
buy_threshold = 5
sell_and_threshold = 20
sell_or_threshold = 40

# 如果改为 smoothing=3，预期需要调整为：
buy_threshold = 3~8
sell_and_threshold = 15~22
sell_or_threshold = 35~42
```

---

**创建日期**: 2026-01-20
**状态**: 待运行实验
**优先级**: 中
**预计影响**: 可能提升5-15%收益（高波动股）
