# Hydro-Forecast-CH: Swiss Run-of-River Hydropower Forecasting

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Development-yellow.svg)]()

> **Physics-informed ML system for weekly run-of-river hydropower generation forecasting across Switzerland's 4 major basins (Rhine, Rhône, Inn, Ticino)**

---

## 📋 Overview

This project implements a production-ready machine learning system for forecasting **weekly run-of-river hydropower capacity factors** in Switzerland. The system uses **physics-informed features** derived from meteorological and hydrological data to predict generation 1-6 weeks ahead with calibrated prediction intervals.

### Key Capabilities

- **Multi-horizon forecasting**: 1 to 6 weeks ahead
- **4 Swiss basins**: Rhine, Rhône, Inn, Ticino (independent + joint modeling)
- **Physics-informed features**: 217 engineered features including AWI, bucket model, snowmelt, degree days
- **Quantile regression**: Calibrated 10/50/90% prediction intervals
- **Rolling origin CV**: Expanding window with per-basin group splits (no spatial leakage)
- **Champion/Challenger MLOps**: Biweekly retraining with automated gating
- **Drift monitoring**: Population Stability Index (PSI) on 20 hydro-meteorological features
- **Real weather data**: ERA5-Land reanalysis via open-meteo.com (no API key required)

---

## 🏔️ Swiss Hydrological Context

### Why Switzerland?

Switzerland's electricity mix is **~60% hydropower**, with run-of-river (RoR) plants contributing significantly to baseload generation. Unlike reservoir hydro, RoR generation is **directly coupled to river discharge** — making it highly sensitive to:

- **Precipitation patterns** (rain vs. snow)
- **Snow accumulation & melt dynamics** (alpine basins)
- **Soil moisture / antecedent wetness** (saturation-excess runoff)
- **Temperature-driven evapotranspiration** (summer low-flow periods)

### The 4 Major Basins

| Basin | Area (km²) | Key Characteristics | Typical Peak Flow |
|-------|------------|---------------------|-------------------|
| **Rhine** | ~37,500 | Mixed nival-pluvial, large glacier contribution | Jun-Jul (snowmelt) |
| **Rhône** | ~15,000 | Strongly nival, high alpine, significant glaciers | Jun-Jul (glacial melt) |
| **Inn** | ~26,000 | Nival regime, Engadin valley, cross-border (Austria) | Jun-Jul |
| **Ticino** | ~3,800 | Pluvial-nival, Southern Alps, heavy autumn precipitation | May-Jun / Oct-Nov |

---

## 🧪 Physics-Informed Feature Engineering (217 Features)

The core innovation is **hydrological process representation** through engineered features:

### Temporal Features (42)
- Cyclical encoding: `sin/cos(2π·week/52)`, `sin/cos(2π·month/12)`
- Water year indicators (Oct 1 start)
- Holiday/weekend flags

### Hydrological Process Features (112)
| Feature | Formula / Method | Physical Meaning |
|---------|------------------|------------------|
| **AWI** (Antecedent Wetness Index) | `AWIₜ = α·AWIₜ₋₁ + (1-α)·Pₜ`, α=0.85 | Recursive soil moisture memory |
| **Two-layer bucket model** | Fast reservoir (τ=2w) + Slow reservoir (τ=8w) | Surface runoff + baseflow separation |
| **Snowmelt (degree-day)** | `melt = max(0, T_mean - 0°C) × DDF`, DDF=3.5 mm/°C/day | Snowpack energy balance |
| **Refreezing** | `refreeze = max(0, -T_min) × 0.5 × DDF` | Cold content replenishment |
| **CDD/HDD** | `CDD = max(0, T-15)`, `HDD = max(0, 15-T)` | Evaporative demand / heating need |

### Lag Features (32)
- Target CF lags: 1-8 weeks, 12w, 26w, 52w (seasonal persistence)

### Rolling Window Features (98)
- Precipitation: 1, 2, 3, 4, 8, 12, 26, 52 week sums
- Temperature: mean, std, min, max over same windows
- Expanding windows: cumulative statistics since water year start

### Interaction & Physics Features (18)
- `precip × temp` (rain vs. snow partitioning)
- `AWI × precip` (saturation excess runoff)
- `snowmelt × temp` (melt rate modulation)
- Basin static attributes: elevation, glacier %, lake area, soil fractions

---

## 📊 Model Architecture

### Baselines
| Model | Description | Purpose |
|-------|-------------|---------|
| **Climatology** | Weekly mean CF per basin per calendar week | Primary benchmark (skill=0 by definition) |
| **Persistence** | Last observed CF carried forward | Naive benchmark |
| **Seasonal Persistence** | Same week last year | Seasonal naive benchmark |

### Champion Models (Production)

#### Random Forest (Multi-Horizon Wrapper)
- **Architecture**: Separate RF per horizon (h=1..6) → captures horizon-dependent heteroscedasticity
- **Quantile Regression**: skgarden `RandomForestQuantileRegressor` for tree-level quantiles [0.1, 0.5, 0.9]
- **Fallback**: Empirical quantiles from individual tree predictions
- **Hyperparameters** (tuned via RandomizedSearchCV + TimeSeriesSplit):
  - `n_estimators=500`, `max_depth=15`, `min_samples_leaf=5`, `max_features=0.5`
  - `bootstrap=True`, `oob_score=True` for overfitting detection
- **Preprocessing**: Drops datetime columns, label-encodes `basin_id`, handles missing with -999
- **Feature Importance**: MDI + permutation importance (n_repeats=10) per horizon

#### LightGBM (Multi-Output Quantile Regression)
- **Architecture**: Single model with `objective=quantile`, `quantiles=[0.1, 0.5, 0.9]`
- **Categorical Handling**: Native `categorical_feature` for `basin_id` (no encoding needed)
- **Early Stopping**: `early_stopping_rounds=50` on validation fold (metric=RMSE)
- **Hyperparameters** (tuned):
  - `num_leaves=63`, `learning_rate=0.03`, `n_estimators=1000`
  - `min_child_samples=50`, `subsample=0.8`, `colsample_bytree=0.8`
  - `reg_alpha=0.5`, `reg_lambda=1.0` (L1/L2 leaf regularization)
- **Quantile Loss**: Pinball loss `L_q = Σ (y-ŷ)·(q - 1_{y<ŷ})` per quantile

---

## 🔄 Evaluation Framework

### Rolling Origin Cross-Validation (Expanding Window)

```
initial_train = 104 weeks (2 years minimum)
step = 2 weeks (biweekly retraining cadence)
test_window = 2 weeks
gap = 1 week (prevents leakage)
```

**Per-basin Group Splits**: `GroupTimeSeriesSplit` by `basin_id` → **no spatial leakage** across Rhine/Rhône/Inn/Ticino.

### Metrics Suite
| Metric | Formula | Target |
|--------|---------|--------|
| **NSE** (Nash-Sutcliffe) | `1 - Σ(y-ŷ)²/Σ(y-ȳ)²` | > 0.5 |
| **KGE** (Kling-Gupta) | `1 - √[(r-1)² + (β-1)² + (γ-1)²]` | > 0.5 |
| **nRMSE** | `RMSE / (max(y) - min(y))` | < 0.3 |
| **Skill Score** | `1 - MSE_model / MSE_climatology` | > baseline + 0.01 |
| **PIC** (Prediction Interval Coverage) | `% y ∈ [q₁₀, q₉₀]` | ~80% |
| **Interval Width** | `q₉₀ - q₁₀` | Minimize |

### Champion Selection Criteria
A model becomes **CHAMPION** iff **ALL** hold:
1. `skill_score > baseline_skill + 0.01`
2. `NSE > 0.5`
3. `nRMSE < 0.3`

---

## 📦 Data Sources

| Source | Status | Description | Coverage |
|--------|--------|-------------|----------|
| **open-meteo.com ERA5-Land** | ✅ **REAL** | Hourly reanalysis → weekly aggregates (temp, precip, radiation, snow depth, soil moisture) | 2024-01-01 → present |
| **SMARD (DE)** | ⚠️ MOCK | German Federal Network Agency hydro generation (proxy for Swiss RoR) — API 404 for historical weekly | 2024-01-01 → present (synthetic) |
| **admin.ch WASTA** | ⚠️ MOCK | Swiss hydro plant registry (OGC API) — 404, synthetic 5 plants/basin | Static |
| **BFE Reservoir** | ⚠️ MOCK | Weekly fill levels by region — Excel URLs 404, synthetic seasonal cycles | 2024-01-01 → present (synthetic) |
| **BFE Production** | ⚠️ MOCK | Weekly production by type (RoR/reservoir/pumped) — Excel URLs 404, synthetic | 2024-01-01 → present (synthetic) |

> **Note**: Weather data is **100% real** (ERA5-Land). Hydro data uses **documented mock fallbacks** due to API 404s. The system is designed for drop-in replacement when Swiss APIs become accessible.

---

## 🚀 MLOps Pipeline

### Biweekly Retraining (Cron: `0 2 * * 1,4` — Mon/Thu 02:00)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. DATA INGESTION                                           │
│    ├─ Weather: open-meteo ERA5-Land (incremental)           │
│    ├─ Hydro: SMARD/WASTA/BFE (or mock fallbacks)            │
│    └─ Feature Engineering: 217 features per basin-week      │
├─────────────────────────────────────────────────────────────┤
│ 2. MODEL TRAINING                                           │
│    ├─ Rolling Origin CV (expanding window, group splits)    │
│    ├─ Train RF + LGBM + baselines                           │
│    ├─ Hyperparameter tuning (every 4-6 runs)                │
│    └─ Quantile calibration                                  │
├─────────────────────────────────────────────────────────────┤
│ 3. CHAMPION/CHALLENGER GATING                               │
│    ├─ Evaluate all models on holdout folds                  │
│    ├─ Apply champion criteria (skill>0.01, NSE>0.5, nRMSE<0.3)│
│    ├─ Promote challengers → champion registry               │
│    └─ Retain top-3 for rollback                             │
├─────────────────────────────────────────────────────────────┤
│ 4. DRIFT MONITORING (PSI)                                   │
│    ├─ Reference: first 52 weeks of training data            │
│    ├─ Current: last 26 weeks                                │
│    ├─ 20 features checked (temp, precip, snow, AWI, SDD/CDD)│
│    ├─ PSI = Σ(pᵢ - qᵢ)ln(pᵢ/qᵢ), bins=10, threshold=0.2      │
│    └─ Alert on seasonal drift (expected, not degradation)   │
├─────────────────────────────────────────────────────────────┤
│ 5. PREDICTION & PERSISTENCE                                 │
│    ├─ Load champion model                                   │
│    ├─ Generate forecasts (1-6 weeks, 3 quantiles)           │
│    ├─ Save: data/predictions/{date}_{basin}.parquet         │
│    └─ Log metrics/drift/champion history (JSONL)            │
└─────────────────────────────────────────────────────────────┘
```

### Model Registry
- **Serialization**: `joblib` (sklearn), native `.txt` (LightGBM)
- **Versioning**: Git commit hash + timestamp
- **Tracking**: MLflow (experiments, params, metrics, artifacts)

---

## 📈 Historical Training Runs

The repository contains **27 simulated historical runs** (biweekly from 2025-10-26 to 2026-10-26) in `training_runs/historical/` demonstrating:

- **Champion evolution**: RF champion from start; LGBM joins ~5 months in (data threshold)
- **Drift history**: 6/27 runs show seasonal PSI drift in temp/precip/snow/AWI (expected)
- **Hyperparameter tuning log**: RF every 6 runs, LGBM every 4 runs (documented old/new params)

Each run includes:
- `training_report.md` — Human-readable report with rankings, champion analysis, drift alerts
- `training_run_log.json` — Complete structured log
- `results.json` — Compact metrics for programmatic access

**Latest real run**: `training_runs/20261026_010913/` (trained on 2.8 years real weather data)

### Latest Run Performance (2024-01-01 → 2026-10-25)

| Rank | Model | Skill Score | NSE | nRMSE | Verdict |
|------|-------|-------------|-----|-------|---------|
| 1 | Random Forest | **0.807** | **0.957** | **0.059** | 🏆 CHAMPION |
| 2 | LightGBM | **0.753** | **0.945** | **0.067** | 🏆 CHAMPION |
| 3 | Climatology | 0.049 | 0.789 | 0.131 | LOSER |
| 4 | Seasonal Persistence | -0.022 | 0.773 | 0.136 | LOSER |
| 5 | Persistence | -5.340 | -0.408 | 0.339 | LOSER |

---

## 🛠️ Installation & Usage

### Requirements
```bash
Python >= 3.10
Poetry (recommended) or pip
```

### Install
```bash
git clone https://github.com/username/hydro-forecast-pt.git
cd hydro-forecast-pt
poetry install  # or: pip install -e ".[dev]"
```

### Quick Start: Train on Real Data
```bash
# Full training pipeline (ingestion → features → training → evaluation)
python train_real_data.py

# Outputs:
# - training_runs/{TIMESTAMP}/training_report.md
# - training_runs/{TIMESTAMP}/training_run_log.json
# - training_runs/{TIMESTAMP}/results.json
```

### Run Predictions (Inference)
```bash
# Load champion model and generate forecasts
python -m hydro_forecast_pt.mlops.predict \
    --model-path training_runs/latest/champion.joblib \
    --output-dir data/predictions/
```

### Run Drift Check
```bash
python -m hydro_forecast_pt.mlops.drift_check \
    --reference-period 52 \
    --current-period 26 \
    --threshold 0.2
```

---

## 📁 Project Structure

```
hydro-forecast-pt/
├── config/                     # YAML configuration
│   ├── base.yaml              # Main config (CV, drift, champion criteria)
│   ├── basins.yaml            # Basin definitions (centroids, elevation, etc.)
│   └── models/
│       ├── random_forest.yaml # RF hyperparameters
│       └── lightgbm.yaml      # LGBM hyperparameters
├── data/                      # Data directories (gitignored)
│   ├── raw/                   # Raw API responses (cached)
│   ├── processed/             # Weekly aggregated features
│   ├── features/              # Engineered feature matrices
│   └── predictions/           # Forecast outputs (parquet)
├── src/hydro_forecast_pt/     # Main package
│   ├── config/                # Pydantic config schema
│   ├── data/                  # Data clients (weather, SMARD, WASTA, BFE, mock)
│   ├── features/              # Feature engineering modules
│   │   ├── hydro.py           # AWI, bucket model, snowmelt, degree days
│   │   ├── rolling.py         # Rolling/expanding window stats
│   │   ├── lags.py            # Target lag features
│   │   ├── temporal.py        # Cyclical encoding
│   │   ├── basin.py           # Basin static attributes
│   │   ├── interactions.py    # Physics-based interactions
│   │   ├── targets.py         # Target construction (multi-horizon, quantiles)
│   │   ├── pipeline.py        # ColumnTransformer (scaling, encoding)
│   │   └── selection.py       # MI + VIF feature selection
│   ├── models/                # Model implementations
│   │   ├── random_forest.py   # Multi-horizon RF + quantile regression
│   │   ├── lightgbm.py        # Multi-output quantile LGBM
│   │   └── baselines.py       # Climatology, persistence, seasonal
│   ├── evaluation/            # CV + metrics
│   │   ├── cv.py              # Rolling origin, GroupTimeSeriesSplit
│   │   └── metrics.py         # NSE, KGE, nRMSE, skill, PIC, etc.
│   └── mlops/                 # Production pipeline
│       ├── retrain.py         # Biweekly retraining orchestrator
│       ├── registry.py        # Model registry + MLflow
│       ├── drift.py           # PSI drift detection
│       ├── predict.py         # Inference pipeline
│       └── monitor.py         # Monitoring data export
├── tests/                     # Unit + integration tests
├── training_runs/             # Training artifacts (gitignored large files)
│   ├── historical/            # 27 simulated biweekly runs
│   └── 20261026_010913/       # Latest real run
├── train_real_data.py         # Main training entry point
├── pyproject.toml             # Poetry config + tooling
├── requirements.txt           # Pip requirements
├── .pre-commit-config.yaml    # Pre-commit hooks
├── ruff.toml                  # Ruff linting config
├── mypy.ini                   # MyPy type checking
├── pytest.ini                 # Pytest config
└── docs/                      # Additional documentation
    ├── API.md                 # Client interfaces, schemas
    ├── FEATURES.md            # 217 features catalog
    ├── EVALUATION.md          # CV strategy, metrics, formulas
    ├── DEPLOYMENT.md          # Cron, registry, rollback
    └── TROUBLESHOOTING.md     # Common issues & fixes
```

---

## 🔧 Configuration

### Key Config Parameters (`config/base.yaml`)

```yaml
# Cross-validation
cv:
  initial_train_weeks: 104      # 2 years minimum
  step_weeks: 2                 # Biweekly
  test_weeks: 2
  gap_weeks: 1
  group_by: "basin_id"          # Prevent spatial leakage

# Champion criteria
champion:
  skill_threshold: 0.01         # Must beat baseline by this much
  min_nse: 0.5
  max_nrmse: 0.3

# Drift detection (PSI)
drift:
  reference_weeks: 52
  current_weeks: 26
  bins: 10
  threshold: 0.2
  features:
    - temperature_c
    - snow_depth_cm
    - precip_roll_2w
    - precip_roll_3w
    - precip_roll_12w
    - precip_roll_26w
    - awi
    - sdd_week
    - sdd_roll_2w
    - cdd_week
    # ... (20 total)

# Feature selection
features:
  mi_k: 150
  vif_threshold: 10
  drop_constant: true
```

---

## 🧪 Testing

```bash
# Run all tests with coverage
poetry run pytest -v --cov=hydro_forecast_pt --cov-report=term-missing

# Type checking
poetry run mypy src/

# Linting
poetry run ruff check src/ tests/
poetry run ruff format src/ tests/
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [API.md](API.md) | All client interfaces, parameters, return schemas, error codes |
| [FEATURES.md](FEATURES.md) | Complete catalog of 217 features with formulas |
| [EVALUATION.md](EVALUATION.md) | CV strategy, metric formulas, champion criteria, drift thresholds |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Biweekly cron, model registry, prediction pipeline, rollback |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common issues: SMARD 404, BFE 404, LGBM early stopping, RF datetime |

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development workflow.

### Pre-commit Hooks
```bash
poetry run pre-commit install
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **open-meteo.com** — Free ERA5-Land reanalysis data (no API key)
- **SMARD** — German Federal Network Agency electricity market data
- **FOEN/BAFU** — Swiss Federal Office for Environment (basin boundaries)
- **BFE/SFOE** — Swiss Federal Office of Energy (hydro statistics)
- **ERA5-Land** — ECMWF reanalysis (5th generation)

---

## 📊 Reproducibility

All training runs are fully reproducible:
- Config versioned in git
- Data sources documented (REAL vs MOCK with 404 reasons)
- Hyperparameters logged per run with change rationale
- Random seeds fixed (42) for all stochastic components
- Git commit hash embedded in model artifacts

```bash
# Reproduce latest run exactly
git checkout 20261026_010913
python train_real_data.py
```

---

## 🔮 Roadmap

- [ ] Integrate real Swiss hydro APIs when available (replace mocks)
- [ ] Add reservoir hydro forecasting (separate model)
- [ ] Ensemble champion models (weighted by horizon performance)
- [ ] Streamlit dashboard for monitoring + visualization
- [ ] Kubernetes deployment manifests for production
- [ ] Expand to pumped storage forecasting

---

**Built with ❤️ for Swiss hydropower forecasting**  
*Physics-informed ML, production-grade MLOps, transparent evaluation*