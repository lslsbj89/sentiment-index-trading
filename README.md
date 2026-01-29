# Sentiment Index Trading System

基于情绪指数的量化交易系统，支持美股七强和加密货币 (BTC, ETH, SOL)。

## 项目结构

```
sentiment-index-trading/
├── stock_strategy/          # 美股情绪策略实验
│   ├── exp_*/               # 各类实验目录
│   └── visualize_*.py       # 可视化工具
├── crypto_strategy/         # 加密货币情绪策略
│   ├── crypto_backtest.py   # 回测脚本
│   └── CRYPTO_BACKTEST_REPORT.md
├── production/              # 生产版本
│   ├── stock/               # 美股生产 (季度更新)
│   └── crypto/              # 加密货币生产 (季度更新)
└── docs/                    # 文档
    ├── WORK_PROGRESS_SUMMARY.md
    └── 2026_THRESHOLD_STAGED_RECOMMENDATION.md
```

## 核心策略

### 买入策略 (v3.0 间隔分批)
- 情绪 < 阈值 → 进入买入模式，买入 20%
- 每隔 7 天 → 再买 20%
- 情绪回升 → 买入全部剩余

### 卖出策略 (双条件)
- **OR条件**: 情绪 > OR阈值 → 无条件卖出
- **AND条件**: 情绪 > AND阈值 且 价格 < MA50 → 卖出

## 资产配置

### 美股七强
| 股票 | 买入阈值 | 最优阈值 |
|------|---------|---------|
| NVDA | 0 | 0 |
| TSLA | 0 | -15 |
| AAPL | 0 | 5 |
| MSFT | 0 | 0 |
| GOOGL | 0 | -5 |
| AMZN | 0 | 0 |
| META | 0 | 0 |

### 加密货币
| 币种 | 买入阈值 | 训练周期 |
|------|---------|---------|
| BTC | <10 | 3年 |
| ETH | <-10 | 3年 |
| SOL | <-10 | 2年 |

## 快速开始

### 美股日常检查
```bash
cd production/stock
python3 retrain_params.py    # 查看当前信号
```

### 加密货币日常检查
```bash
cd production/crypto
python3 retrain_params.py    # 查看当前信号
```

### 季度参数更新
```bash
# 每季度 (1月/4月/7月/10月) 运行
cd production/stock && python3 retrain_params.py
cd production/crypto && python3 retrain_params.py
```

## 数据源

- **美股情绪**: Fear Greed Index S3 (Smoothing=3, MoneyFlow=13)
- **加密货币情绪**: yahoo_artemis_index (Smoothing=3)
- **价格数据**: PostgreSQL (`crypto_fear_greed_2` 数据库)

## 回测结果

### 美股 (2019-2025, 7窗口 Walk-Forward)
- NVDA: +1510.7%
- TSLA: +413.3%
- AAPL: +310.0%
- 合计: $3,776K (初始 $700K)

### 加密货币 (2023-2025)
- BTC: +344.3%
- ETH: +90.3%
- SOL: +1195.4%
- 合计: $1,930K (初始 $300K)

## 许可证

MIT License

---

*从 [AAPL](https://github.com/lslsbj89/AAPL) 仓库分离 (v1.0-unified tag)*
