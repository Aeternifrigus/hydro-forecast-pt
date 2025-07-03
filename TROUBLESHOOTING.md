# Troubleshooting Guide

Common issues, root causes, and resolutions for the Hydro-Forecast-CH system.

---

## Quick Reference

| Symptom | Likely Cause | Quick Fix |
|---------|--------------|-----------|
| `SMARD_404` in logs | Historical weekly endpoint deprecated | Auto-fallback to `MockSMARDClient` |
| `WASTA_404` in logs | OGC API endpoint changed/removed | Auto-fallback to `MockWASTAClient` |
| `BFE_404` in logs | Excel URLs return 404 | Auto-fallback to `MockBFEClient` |
| LightGBM early stopping error | Insufficient training data for CV folds | Increase `initial_train_weeks` to 104 |
| RF "datetime column" error | Categorical encoder receives datetime | Use `_preprocess_features()` to drop/encode |
| Drift alerts on every run | Seasonal features (temp, snow) always drift | Expected — review but don't block |
| CV produces < 3 folds | Training window too short | Set `initial_train_weeks >= 104` |
| Out of memory on training | Too many features + trees | Reduce `max_features`, `n_estimators` |

---

## Data Ingestion Issues

### SMARD API Returns 404 for Historical Weekly Data

**Error:**
```
SMARD_404: GET https://smard.de/app/chart_data/4007/DE/week/2025-01-01/2025-01-31.json 404 Not Found
```

**Root Cause:** SMARD (German Federal Network Agency) deprecated their historical weekly aggregation endpoints. The quarter-hourly endpoints still work but require massive downloads (~2M rows/year).

**Resolution:** System automatically falls back to `MockSMARDClient` which generates realistic German run-of-river capacity factors with:
- Seasonal pattern: summer high (0.6-0.8), winter low (0.2-0.4)
- Weekly autocorrelation (AR1, φ=0.85)
- Basin-specific noise per Rhine/Rhône/Inn/Ticino

**Code Location:** `src/hydro_forecast_pt/data/smard_client.py:63-80`

```python
# Automatic fallback logic
try:
    return self._fetch_real_data(...)
except HTTPError as e:
    if e.response.status_code == 404:
        logger.warning("SMARD_404: Using MockSMARDClient fallback")
        return MockSMARDClient().fetch_generation(...)
    raise
```

**Verification:**
```bash
# Check logs for fallback activation
grep "SMARD_404" logs/retrain.log
# Should see: "WARNING - SMARD_404: Using MockSMARDClient fallback"
```

---

### WASTA OGC API Returns 404

**Error:**
```
WASTA_404: GET https://api.geo.admin.ch/wasta/ogc/features/collections/plants/items 404 Not Found
```

**Root Cause:** The Swiss Federal Office of Energy (BFE) migrated the WASTA plant registry from OGC API to a new GeoPackage download. The old endpoint is gone.

**Resolution:** System uses `MockWASTAClient` with 5 realistic plants per basin:
- Coordinates from actual Swiss hydropower locations
- Types: run_of_river, reservoir, pumped_storage
- Capacities: 10-500 MW range
- Basin assignment via Shapely point-in-polygon

**Mock Data (per basin):**
| Basin | Plant | Type | Capacity (MW) | Coordinates |
|-------|-------|------|---------------|-------------|
| Rhine | Albbruck-Dogern | RoR | 72 | 47.56, 8.13 |
| Rhine | Rheinfelden | RoR | 120 | 47.55, 7.79 |
| Rhône | Chancy-Pougny | RoR | 85 | 46.15, 5.95 |
| Rhône | Verbois | RoR | 110 | 46.19, 6.04 |
| Inn | Pradella | RoR | 110 | 46.78, 10.07 |
| Ticino | Piottino | RoR | 65 | 46.55, 8.78 |

**Code Location:** `src/hydro_forecast_pt/data/wasta_client.py:45-70`

---

### BFE Excel URLs Return 404

**Error:**
```
BFE_404: GET https://www.bfe.admin.ch/bfe/en/home/supply/statistics/weekly-reservoir-fill.xlsx 404 Not Found
```

**Root Cause:** BFE (Swiss Federal Office of Energy) restructured their publication URLs. The weekly reservoir fill and production Excel files moved to a new CMS with dynamic URLs.

**Resolution:** `MockBFEClient` generates physically plausible data:
- **Reservoir fill**: Sinusoidal annual cycle (winter ~30%, summer ~85%) + AR1 noise
- **Production by type**: RoR ~55%, Reservoir ~35%, Pumped ~10% of Swiss total
- Weekly resolution matching operational reporting

```python
# MockBFEClient seasonal pattern
def _seasonal_fill(self, week: int) -> float:
    # Peak week 30 (late July), trough week 52/1 (Jan)
    phase = 2 * np.pi * (week - 30) / 52
    return 0.575 + 0.275 * np.cos(phase) + np.random.normal(0, 0.05)
```

**Code Location:** `src/hydro_forecast_pt/data/bfe_client.py:55-85`

---

## Model Training Issues

### LightGBM Early Stopping Failure

**Error:**
```
LGBM_EARLY_STOP: Early stopping triggered after 0 iterations.
ValueError: Training data has only 1 fold but early_stopping_rounds requires at least 2 validation folds.
```

**Root Cause:** Initial configuration used `initial_train_weeks=26` (6 months). With biweekly step (2w) and test (2w), this yields only 1 CV fold — insufficient for early stopping which needs validation data.

**Resolution:** Increased `initial_train_weeks` from 26 → 104 (2 years minimum).

```yaml
# config/base.yaml - FIXED
cv:
  initial_train_weeks: 104  # Was: 26
  step_weeks: 2
  test_weeks: 2
  gap_weeks: 1
```

**Mathematical Justification:**
- Need ≥ 3 folds for reliable early stopping
- Folds = floor((total_weeks - initial_train) / step) - 1
- With 144 weeks data (2024-01 to 2026-10):
  - initial_train=26 → (144-26)/2 = 59 → 58 folds ✓ (but first fold too small)
  - initial_train=104 → (144-104)/2 = 20 → 19 folds ✓ (each fold has 104+ weeks)

**Commit:** `feat(cv): increase initial_train to 104w for LGBM early stopping (fixes #12)`

---

### RandomForest Datetime Column Error

**Error:**
```
TypeError: Cannot convert column 'date' to numeric. 
ColumnTransformer received datetime64[ns] columns: ['date', 'week_start']
```

**Root Cause:** `ColumnTransformer` with `StandardScaler` received raw datetime columns from feature engineering pipeline. The preprocessor expected only numeric/categorical.

**Resolution:** Added `_preprocess_features()` method to `RandomForestModel`:

```python
def _preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
    """Drop datetime columns, label-encode basin_id."""
    X = X.copy()
    
    # Drop datetime-like columns
    datetime_cols = X.select_dtypes(include=['datetime64', 'datetimetz']).columns
    X = X.drop(columns=datetime_cols)
    
    # Label encode basin_id (categorical)
    if 'basin_id' in X.columns:
        X['basin_id'] = self._basin_encoder.transform(X['basin_id'])
    
    return X
```

**Code Location:** `src/hydro_forecast_pt/models/random_forest.py:45-62`

**Commit:** `fix(rf): add _preprocess_features to drop datetime cols and encode basin_id`

---

### Cross-Validation Produces Insufficient Folds

**Error:**
```
CV_INSUFFICIENT_DATA: Only 1 fold generated. Need at least 3 for reliable CV.
Consider increasing initial_train_weeks or extending data range.
```

**Root Cause:** Data range too short relative to `initial_train_weeks + gap + test`.

**Resolution:** Ensure data spans sufficient period:
- Minimum: `initial_train_weeks + 3 * (step_weeks + test_weeks + gap_weeks)`
- With defaults (104 + 2 + 2 + 1 = 109): need ~115 weeks ≈ 2.2 years
- Our data: 2024-01-01 to 2026-10-25 = ~144 weeks ✓

**If using shorter data:**
```yaml
# Temporarily reduce for testing
cv:
  initial_train_weeks: 52  # 1 year minimum for testing only
  step_weeks: 2
  test_weeks: 2
  gap_weeks: 1
```

---

## Drift Detection Issues

### Drift Alerts on Every Retraining Run

**Symptom:** PSI drift detected for 8-12 features every biweekly run.

**Features typically flagged:**
```
temperature_c: PSI=0.234
snow_depth_cm: PSI=0.229
cdd_week: PSI=0.267
hdd_week: PSI=0.312
snowmelt_mm: PSI=0.245
precip_roll_26w: PSI=0.201
```

**Root Cause:** **This is expected behavior.** These features have strong annual cycles. The reference period (first 52 weeks) and current period (most recent 26 weeks) fall in different seasons → distribution shift.

**Why This Happens:**
- Reference: Weeks 1-52 of training (e.g., Jan 2024 - Dec 2024)
- Current: Weeks 119-144 (e.g., May 2026 - Oct 2026)
- Temperature in Jan vs Jul → completely different distributions

**Resolution:** **Do not block retraining on seasonal drift.**
- Cyclical encoding (`sin_week`, `cos_week`) is drift-invariant
- Expanding window retraining always includes full seasonal cycles
- Drift alerts are logged for awareness but don't auto-reject models

**Code Handling:**
```python
# In RetrainPipeline.run()
drift_results = check_drift(X, config.drift)
# Log but don't fail
if drift_results["has_drift"]:
    logger.warning(f"Seasonal drift detected: {drift_results['alerts']}")
    # Continue — champion selection proceeds normally
```

**When to Worry:** Non-seasonal features drifting (e.g., `basin_elevation_mean_m`, `awi` in same season) → indicates data pipeline issue or real covariate shift.

---

### PSI Threshold Too Sensitive/Insensitive

**Symptom:** Too many false alerts OR missing real drift.

**Tuning:**
```yaml
drift:
  threshold: 0.2        # Default: standard PSI threshold
  bins: 10              # Deciles of reference distribution
  reference_weeks: 52   # Must cover full seasonal cycle
  current_weeks: 26     # Recent window
```

**Adjustment Guide:**
| Situation | Threshold | Bins | Reference |
|-----------|-----------|------|-----------|
| Too many alerts (seasonal) | 0.3-0.5 | 10 | 52w |
| Missing real drift | 0.1-0.15 | 15-20 | 52w |
| Noisy features | 0.25 | 5 | 104w |

---

## Performance Issues

### Training Takes Too Long (> 30 min)

**Profiling:**
```bash
# Add timing to retrain script
python -m cProfile -o profile.stats -m hydro_forecast_pt.mlops.retrain
# Then analyze:
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"
```

**Common Bottlenecks:**

| Bottleneck | Fix |
|------------|-----|
| RF with 500 trees × 217 features | Reduce `n_estimators=300`, `max_features=0.3` |
| LGBM with 1000 iterations | Reduce `n_estimators=500`, increase `learning_rate=0.05` |
| Feature engineering on full history | Use incremental rolling window updates |
| QuantileRF per-horizon training | Use multi-output QuantileRF |

**Optimized Config for Speed:**
```yaml
models:
  random_forest:
    n_estimators: 300
    max_depth: 10
    max_features: 0.3
    n_jobs: -1
  lightgbm:
    n_estimators: 500
    learning_rate: 0.05
    num_leaves: 31
```

---

### Out of Memory (OOM) During Training

**Error:**
```
MemoryError: Unable to allocate array with shape (150000, 217)
# Or: Killed (OOM killer)
```

**Root Cause:** Feature matrix too large for available RAM.

**Resolutions:**

1. **Reduce features** (pre-MI selection):
   ```python
   # In FeaturePipeline
   selector = VarianceThreshold(threshold=1e-5)
   mi_selector = SelectKBest(mutual_info_regression, k=100)  # Was 150
   ```

2. **Chunk training data:**
   ```python
   # Process basins sequentially
   for basin in basins:
       X_basin = X[X['basin_id'] == basin]
       y_basin = y[X['basin_id'] == basin]
       model.fit(X_basin, y_basin)
   ```

3. **Use LGBM (lower memory):**
   ```yaml
   # LGBM uses histogram-based binning, much lower memory
   lightgbm:
     n_estimators: 500
     max_bin: 255  # Reduce from default 255 if needed
   ```

4. **Swap/Cloud:** Run on machine with ≥ 16GB RAM.

---

## Prediction Issues

### Forecasts Look Constant / Flat

**Symptom:** All horizons return same value per basin.

**Root Cause:** Model predicting training mean (underfitting) or feature mismatch.

**Debugging:**
```python
# 1. Check feature alignment
model, metadata = registry.get_champion("random_forest")
print("Expected features:", metadata["feature_names"])
print("Actual features:", list(X_latest.columns))

# 2. Check predictions per horizon
preds = model.predict_quantiles(X_latest)
for q in [0.1, 0.5, 0.9]:
    print(f"q={q}: {preds[q][0]}")  # Should vary by horizon

# 3. Verify basin_id encoding
print(X_latest['basin_id'].unique())  # Should be 0,1,2,3 not strings
```

**Common Fix:** Ensure `_preprocess_features()` applied identically at train and inference.

---

### Prediction Intervals Too Wide / Too Narrow

**Symptom:** PIC (coverage) far from 80% for 0.1/0.9 quantiles.

| PIC | Issue | Fix |
|-----|-------|-----|
| < 60% | Intervals too narrow | Increase `min_samples_leaf` (RF), add `reg_alpha` (LGBM) |
| > 95% | Intervals too wide | Decrease `min_samples_leaf`, reduce `n_estimators` |
| Varies wildly by horizon | Horizon-specific calibration | Fit separate quantile models per horizon |

**Calibration Check:**
```python
from hydro_forecast_pt.evaluation.metrics import prediction_interval_coverage

for h in range(6):
    pic = prediction_interval_coverage(
        y_true[:, h], q_low[:, h], q_high[:, h], alpha=0.1
    )
    print(f"Horizon {h+1}w: PIC={pic:.3f}")
```

---

## Deployment Issues

### Cron Job Not Running

**Check:**
```bash
# 1. Cron daemon running?
systemctl status cron

# 2. Crontab installed?
crontab -l -u hydro-forecast

# 3. Permissions on script?
ls -la /opt/hydro-forecast/venv/bin/python

# 4. Log output?
tail -f /var/log/hydro-forecast/retrain.log
```

**Common Issue:** Cron uses minimal PATH. Use absolute paths:
```bash
# WRONG
0 2 * * 1,4 python -m hydro_forecast_pt.mlops.retrain

# CORRECT
0 2 * * 1,4 /opt/hydro-forecast/venv/bin/python -m hydro_forecast_pt.mlops.retrain \
    --config /opt/hydro-forecast/config/base.yaml
```

---

### Model Registry Corrupted / Missing

**Symptom:** `FileNotFoundError: models/registry/random_forest/champion/model.joblib`

**Recovery:**
```bash
# 1. List available versions
python -m hydro_forecast_pt.mlops.registry list-challengers random_forest

# 2. Promote latest challenger
python -m hydro_forecast_pt.mlops.registry promote \
    --model random_forest \
    --version v20261011_010913

# 3. Or retrain from scratch
python -m hydro_forecast_pt.mlops.retrain --force-retrain true
```

**Prevention:** Enable daily backups (see DEPLOYMENT.md).

---

### GitHub Actions Workflow Fails

**Common Failures:**

| Error | Fix |
|-------|-----|
| `poetry: command not found` | Add `pip install poetry` step |
| `Permission denied (push)` | Use `GITHUB_TOKEN` with `contents: write` |
| `timeout` (60 min default) | Add `timeout-minutes: 90` |
| `MLflow tracking URI` | Use `sqlite:///mlflow.db` local path |

**Working Workflow Template:** See DEPLOYMENT.md → GitHub Actions section.

---

## Debugging Checklist

### When Retraining Fails

```bash
# 1. Check data freshness
ls -la data/raw/weather/  # Should have recent .nc files

# 2. Verify config
cat config/base.yaml | grep -A 20 "cv:"

# 3. Run with debug logging
python -m hydro_forecast_pt.mlops.retrain --verbose 2>&1 | tee debug.log

# 4. Check specific stage
python -c "
from hydro_forecast_pt.data.pipeline import DataPipeline
p = DataPipeline(config, Path('data'))
d = p.run(date(2024,1,1), date(2026,10,25))
print('Features:', d['features'].shape)
print('Targets:', d['targets'].shape)
print('NaN features:', d['features'].isna().sum().sum())
"
```

### When Drift Alerts Seem Wrong

```bash
# 1. Inspect PSI per feature
python -c "
from hydro_forecast_pt.mlops.drift import PSIDriftDetector
detector = PSIDriftDetector(ref_data, drift_features)
psi = detector.compute_psi(curr_data)
for f, v in sorted(psi.items(), key=lambda x: -x[1]):
    print(f'{f}: {v:.4f}')
"

# 2. Plot distributions
python -c "
import matplotlib.pyplot as plt
for f in ['temperature_c', 'awi', 'snowmelt_mm']:
    plt.figure()
    plt.hist(ref_data[f], bins=20, alpha=0.5, label='ref')
    plt.hist(curr_data[f], bins=20, alpha=0.5, label='curr')
    plt.title(f); plt.legend(); plt.savefig(f'drift_{f}.png')
"
```

---

## Version-Specific Fixes

### pandas 2.x / xarray Compatibility

**Issue:** `xarray.open_dataset()` fails with netCDF4 engine on pandas 2.x.

**Fix:** Explicit engine and decode settings:
```python
# In WeatherClient
ds = xarray.open_dataset(
    cache_path, 
    engine='netcdf4',           # Required for pandas 2.x
    decode_times=True,          # Ensure proper datetime decoding
    chunks={'time': 1000}       # Chunk for memory
)
```

**Commit:** `fix(weather): pandas 2.x compat - engine='netcdf4', decode_times=True`

---

### scikit-learn 1.3+ ColumnTransformer API Change

**Issue:** `ColumnTransformer` now requires `verbose_feature_names_out=False` for cleaner feature names.

**Fix:**
```python
preprocessor = ColumnTransformer(
    transformers=[...],
    verbose_feature_names_out=False  # Clean names: "num__temp" not "remainder__temp"
)
```

---

## Getting Help

1. **Check logs first:** `tail -100 logs/retrain.log`
2. **Run minimal reproduction:** Isolate failing component
3. **Check git history:** `git log --oneline --grep="fix" --since="2025-01-01"`
4. **Compare with historical runs:** `ls training_runs/historical/`

**Key Debug Commands:**
```bash
# Full pipeline test with small data
python -m hydro_forecast_pt.mlops.retrain \
    --config config/test.yaml \
    --run-date 2024-06-01 \
    --skip-drift-check true

# Validate model registry
python -m hydro_forecast_pt.mlops.registry list-champions

# Test prediction
python -m hydro_forecast_pt.mlops.predict \
    --model random_forest \
    --forecast-date 2026-10-26 \
    --verify
```