# Crypto 2026 Operation Guide

**Data**: `yahoo_artemis_index` (sentiment, smoothing=3) + `yahoo_candles` (price)

---

## Assets

| Coin | Buy Threshold | Train Period | Reason |
|------|--------------|-------------|--------|
| BTC | sent < 10 | 3 yr | Sentiment stays high in bull market, needs aggressive threshold |
| ETH | sent < -10 | 3 yr | Selective buying on significant dips |
| SOL | sent < -10 | 2 yr | Shorter train avoids 2021 bull skew (100x rally) |

---

## Buy Strategy (fixed, no training needed)

| Parameter | BTC | ETH | SOL |
|-----------|-----|-----|-----|
| Buy trigger | sent < 10 | sent < -10 | sent < -10 |
| Interval | 7 days | 7 days | 7 days |
| Batch size | 20% of base | 20% of base | 20% of base |
| Recovery buy-all | sent >= 10 | sent >= -10 | sent >= -10 |

**Flow**:
1. Sentiment drops below threshold -> enter buy mode, buy 20% of base capital
2. Every 7 days in buy mode -> buy another 20%
3. Sentiment recovers above threshold -> buy ALL remaining cash, exit buy mode
4. After a sell -> base capital = current total value (dynamic compounding)

---

## Sell Strategy

**Dual-condition sell** (either triggers a full sell):
- **OR**: sentiment > OR threshold -> sell unconditionally
- **AND**: sentiment > AND threshold **AND** price < MA50 -> sell

### 2026-Q1 Sell Parameters (train: 2023-2025 / 2024-2025)

| Coin | AND | OR | Train Return | Meaning |
|------|-----|-----|-------------|---------|
| BTC | >20 | >50 | +352.3% | Moderate, wait for clear overheating |
| ETH | >5 | >30 | +148.2% | Sensitive, quick exit on slight euphoria |
| SOL | >15 | >30 | +104.4% | Moderate AND, sensitive OR |

---

## Daily Check

```
1. Get each coin's sentiment (yahoo_artemis_index, smoothed)

2. Holding a position -> Check sell
   Sentiment > OR threshold?          -> Sell all
   Sentiment > AND threshold & < MA50? -> Sell all
   After sell: reset buy state, update base capital

3. No position / in buy mode -> Check buy
   No position & sent < threshold?     -> Enter buy mode, buy 20%
   In buy mode & >= 7 days since last? -> Buy another 20%
   In buy mode & sent >= threshold?    -> Buy ALL remaining, exit buy mode
```

---

## Quarterly Update (every 3 months)

```bash
cd 2026_crypto_production

# Auto (uses latest data, per-coin train length)
python3 retrain_params.py

# Or specify train years manually
python3 retrain_params.py 2023 2025    # Q1
python3 retrain_params.py 2023 2026    # Q2+ (includes 2026 data)
```

**Schedule**:

| Time | BTC/ETH Train | SOL Train | Action |
|------|--------------|----------|--------|
| 2026-01 (done) | 2023-2025 (3yr) | 2024-2025 (2yr) | Current params |
| 2026-04 | 2023-2026 (3yr) | 2024-2026 (2yr) | Retrain with Q1 data |
| 2026-07 | 2023-2026 (3yr) | 2024-2026 (2yr) | Retrain with H1 data |
| 2026-10 | 2024-2026 (3yr) | 2025-2026 (2yr) | Retrain with Q3 data |

After retraining, update the sell parameters table above with new values.

---

## Backtest Reference

Full backtest report: `crypto_strategy/CRYPTO_BACKTEST_REPORT.md`

| Coin | Backtest Return (2023-2025) | Strategy |
|------|---------------------------|----------|
| BTC | +344.3% | Interval buy<10 |
| ETH | +90.3% | Interval buy<-10 |
| SOL | +1195.4% | Interval buy<-10 |
| **Total** | **$1,929,918 / $300K** | **Interval 3:0** |

Key findings:
- Interval buy beats staged buy 3:0 across all coins
- Interval buy training (matching test strategy) adds +9.8% vs staged buy training
- SOL needs 2yr train (3yr includes 2021 100x rally, skews sell params)
- Crypto OR range (30-200) is much wider than US stocks (30-60)

---

## Trading Costs

| Parameter | Value |
|-----------|-------|
| Commission | 0.1% |
| Slippage | 0.1% |
| MA period | 50 days |

## Files

| File | Purpose |
|------|---------|
| `OPERATION_GUIDE.md` | This document |
| `retrain_params.py` | Quarterly parameter retraining |
| `params_*.csv` | Buy + sell parameter outputs |

---

*Last updated: 2026-01-28*
