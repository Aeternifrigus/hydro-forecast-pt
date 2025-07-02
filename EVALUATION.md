# Evaluation Framework

Complete specification of cross-validation strategy, metrics, champion criteria, and drift thresholds.

---

## Cross-Validation Strategy

### Rolling Origin with Expanding Window

```
┌─────────────────────────────────────────────────────────────────┐
│ Time Series Cross-Validation (Rolling Origin)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Fold 1:  [===Train (104w)===] [Gap 1w] [==Test (2w)==]        │
│  Fold 2:  [==========Train (106w)==========] [Gap] [==Test==]  │
│  Fold 3:  [===============Train (108w)==============] [Gap][T]  │
│  ...                                                            │
│  Fold N:  [=========================Train====================]   │
│                                                                 │
│  Parameters:                                                    │
│  - initial_train_weeks: 104  (2 years minimum)                 │
│  - step_weeks: 2            (biweekly retraining cadence)      │
│  - test_weeks: 2            (evaluation window)                │
│  - gap_weeks: 1             (prevents leakage)                 │
│  - group_by: "basin_id"     (spatial split)                    │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Design?

1. **Expanding window** → Uses all historical data (no arbitrary truncation)
2. **Biweekly step** → Matches operational retraining cadence
3. **1-week gap** → Prevents leakage from feature engineering (rolling windows)
4. **Group splits by basin** → No spatial leakage across Rhine/Rhône/Inn/Ticino

### GroupTimeSeriesSplit Implementation

```python
class GroupTimeSeriesSplit:
    """Ensures all samples from a group (basin) in a fold are either 
    ALL train or ALL test — never mixed."""
    
    def split(self, X, y, groups):
        # Sort by time within each group
        # Create time-ordered splits
        # For each fold: assign entire groups to train or test
        # Never split a group across train/test boundary
```

**Number of folds**: ~20-25 (depending on data length after initial_train=104)

---

## Metrics Suite

### Primary Metrics

| Metric | Formula | Range | Interpretation |
|--------|---------|-------|----------------|
| **NSE** (Nash-Sutcliffe) | `1 - Σ(y-ŷ)² / Σ(y-ȳ)²` | (-∞, 1] | >0.5 good, >0.75 very good, 1=perfect |
| **KGE** (Kling-Gupta) | `1 - √[(r-1)² + (β-1)² + (γ-1)²]` | (-∞, 1] | Decomposes into correlation, bias, variability |
| **nRMSE** | `RMSE / (max(y) - min(y))` | [0, ∞) | <0.3 good, <0.1 excellent |
| **Skill Score** | `1 - MSE_model / MSE_climatology` | (-∞, 1] | >0 beats climatology, >0.01 for champion |

### Secondary Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| **MAE** | `mean(|y - ŷ|)` | Robust to outliers |
| **MAPE** | `mean(|y - ŷ| / y) × 100` | Relative error (%) |
| **PBIAS** | `Σ(y - ŷ) / Σ(y) × 100` | Systematic bias (%) |
| **Correlation (r)** | `corr(y, ŷ)` | Linear association |
| **R²** | `1 - Σ(y-ŷ)² / Σ(y-ȳ)²` | Variance explained |

### Prediction Interval Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| **PIC** (Coverage) | `% y ∈ [q₁₀, q₉₀]` | ~80% (for 0.1/0.9) |
| **PICP** | Same as PIC | Per-horizon |
| **MPIW** (Mean Width) | `mean(q₉₀ - q₁₀)` | Minimize |
| **Interval Score** | `(q₉₀ - q₁₀) + 2/α × (q₁₀ - y)₊ + 2/α × (y - q₉₀)₊` | Proper scoring rule |

### Per-Horizon & Per-Basin Breakdown

All metrics computed at multiple granularities:

1. **Aggregate**: All basins × all horizons pooled
2. **Per-horizon**: h=1,2,3,4,5,6 weeks separately
3. **Per-basin**: Rhine, Rhône, Inn, Ticino separately
4. **Per-season**: DJF, MAM, JJA, SON
5. **Per-basin × horizon**: 4 × 6 = 24 combinations

---

## Champion Selection Criteria

### Mathematical Definition

A model achieves **CHAMPION** status iff **ALL** conditions hold:

```
CHAMPION ⇔ (Skill > Baseline_Skill + δ) ∧ (NSE > τ_NSE) ∧ (nRMSE < τ_nRMSE)
```

Where:
- `δ = 0.01` (minimum practical improvement over baseline)
- `τ_NSE = 0.5` (minimum acceptable NSE)
- `τ_nRMSE = 0.3` (maximum acceptable normalized RMSE)

### Rationale

| Criterion | Threshold | Justification |
|-----------|-----------|---------------|
| Skill > baseline + 0.01 | 0.01 | Practical significance; avoids promoting marginal improvements |
| NSE > 0.5 | 0.5 | Hydrological standard: model better than mean predictor |
| nRMSE < 0.3 | 0.3 | Normalized error < 30% of range = useful predictions |

### Decision Matrix

| Model | Skill | NSE | nRMSE | Verdict | Reason |
|-------|-------|-----|-------|---------|--------|
| RF | 0.807 | 0.957 | 0.059 | 🏆 CHAMPION | All criteria met |
| LGBM | 0.753 | 0.945 | 0.067 | 🏆 CHAMPION | All criteria met |
| Climatology | 0.049 | 0.789 | 0.131 | LOSER | Skill not > baseline+0.01 |
| Seasonal Persist | -0.022 | 0.773 | 0.136 | LOSER | Skill < baseline |
| Persistence | -5.340 | -0.408 | 0.339 | LOSER | NSE ≤ 0, nRMSE > 0.3 |

### Challenger Tracking

- Top-3 models retained in registry for rollback
- Challenger = meets skill threshold but not all champion criteria
- Logged with full metrics, hyperparameters, training commit hash

---

## Drift Detection (Population Stability Index)

### PSI Formula

```
PSI = Σᵢ (pᵢ - qᵢ) × ln(pᵢ / qᵢ)
```

Where:
- `pᵢ` = proportion of reference population in bin i
- `qᵢ` = proportion of current population in bin i
- `bins = 10` (deciles of reference distribution)
- `threshold = 0.2` (standard PSI threshold)

### Drift Configuration

```yaml
drift:
  reference_weeks: 52      # First 52 weeks of training data
  current_weeks: 26        # Most recent 26 weeks
  bins: 10                 # Decile binning
  threshold: 0.2           # Alert threshold
  features: 20             # Hydro-meteorological features
```

### Features Monitored (20)

| Feature | Category | Expected Drift Pattern |
|---------|----------|------------------------|
| `temperature_c` | Weather | Strong seasonal (annual cycle) |
| `snow_depth_cm` | Weather | Strong seasonal (winter accumulation) |
| `precip_roll_2w` | Rolling | Seasonal (autumn/winter peaks) |
| `precip_roll_3w` | Rolling | Seasonal |
| `precip_roll_12w` | Rolling | Seasonal |
| `precip_roll_26w` | Rolling | Strong seasonal (half-year) |
| `awi` | Hydro | Seasonal (soil moisture cycle) |
| `sdd_week` | Hydro | Strong seasonal (summer cooling) |
| `sdd_roll_2w` | Hydro | Strong seasonal |
| `cdd_week` | Hydro | Strong seasonal (winter heating) |
| `cdd_roll_4w` | Hydro | Seasonal |
| `soil_moisture_proxy` | Hydro | Seasonal |
| `bucket_storage_mm` | Hydro | Seasonal |
| `snowmelt_mm` | Hydro | Strong seasonal (spring melt) |
| `runoff_proxy` | Hydro | Seasonal |
| `precip_mm` | Weather | Seasonal |
| `radiation_mj_m2` | Weather | Strong seasonal (summer max) |
| `snow_depth_cm` | Weather | Duplicate check |
| `bucket_fast_storage` | Hydro | Seasonal |
| `bucket_slow_storage` | Hydro | Seasonal |

### Drift Interpretation

| PSI Value | Interpretation | Action |
|-----------|----------------|--------|
| < 0.1 | No drift | Continue monitoring |
| 0.1 - 0.2 | Minor drift | Log, investigate |
| 0.2 - 0.5 | Moderate drift | Alert, consider retrain |
| > 0.5 | Major drift | Immediate retrain, investigate |

**Important**: Seasonal features (temperature, snow, degree days) **will always show drift** due to annual cycles. This is **expected behavior**, not model degradation.

### Seasonal Drift Mitigation

1. **Cyclical encoding** (sin/cos week) — invariant to seasonal shift
2. **Expanding window retraining** — always includes full annual cycles
3. **Reference period = 52 weeks** — covers full seasonal cycle
4. **Drift alerts logged but not auto-blocking** — human review required

---

## Evaluation Reporting

### Per-Run Report Structure (`training_report.md`)

```markdown
# Hydro-Forecast-CH Training Report

**Date Range:** 2024-01-01 to 2026-10-25
**Run Date:** 2026-10-26 01:09:13

## Model Performance
| Rank | Model | Skill | NSE | nRMSE | Verdict |
|------|-------|-------|-----|-------|---------|
| 1 | random_forest | 0.807 | 0.957 | 0.059 | CHAMPION |
| 2 | lightgbm | 0.753 | 0.945 | 0.067 | CHAMPION |
| 3 | climatology | 0.049 | 0.789 | 0.131 | LOSER |

## Champion/Challenger Analysis
### random_forest - CHAMPION
- Skill: 0.807 > baseline 0.049 + 0.01
- NSE: 0.957 > 0.5
- nRMSE: 0.059 < 0.3

## Drift Analysis
- Status: DRIFT_DETECTED
- Features checked: 20
- Features with drift: 9 (seasonal, expected)
- Alerts: temperature_c PSI=0.234, snow_depth_cm PSI=0.229, ...

## Hyperparameter Changes
- RF: n_estimators=500→200, max_depth=20→10 (overfitting fix)
- LGBM: num_leaves=31→63, lr=0.03→0.05 (capacity increase)
```

### JSON Log Structure (`training_run_log.json`)

```json
{
  "start_time": "2026-10-26T01:09:13",
  "date_range": {"start": "2024-01-01", "end": "2026-10-25"},
  "data_sources": {...},
  "models": {...},
  "champion_selection": {...},
  "drift_analysis": {...},
  "hyperparameter_changes": {...},
  "end_time": "2026-10-26T01:11:03",
  "status": "completed"
}
```

---

## Historical Run Comparison

| Run Date | Data End | Champions | Drift | Folds |
|----------|----------|-----------|-------|-------|
| 2025-10-26 | 2025-10-19 | RF, LGBM | OK | 1 |
| 2025-11-09 | 2025-11-02 | RF, LGBM | DRIFT | 1 |
| ... | ... | ... | ... | ... |
| 2026-10-25 | 2026-10-18 | RF, LGBM | OK | 3 |

**Champion Evolution**: RF champion from start; LGBM joins ~month 5 when data sufficient for its higher capacity.

---

## Reproducibility Checklist

- [ ] Random seeds fixed (42) for all stochastic components
- [ ] Config versioned in git (config/*.yaml)
- [ ] Data sources documented (REAL vs MOCK with 404 reasons)
- [ ] Hyperparameters logged per run with change rationale
- [ ] Git commit hash embedded in model artifacts
- [ ] Environment pinned (pyproject.toml + poetry.lock)