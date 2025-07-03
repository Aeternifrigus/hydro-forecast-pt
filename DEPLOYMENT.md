# Deployment Guide

Production deployment procedures for the Hydro-Forecast-CH system.

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                        PRODUCTION PIPELINE                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │
│  │   Cron       │───▶│  Retrain     │───▶│  Champion    │         │
│  │  (Mon/Thu    │    │  Pipeline    │    │  Selection   │         │
│  │   02:00)     │    │              │    │              │         │
│  └──────────────┘    └──────┬───────┘    └──────┬───────┘         │
│                             │                   │                  │
│                    ┌────────▼────────┐  ┌───────▼───────┐         │
│                    │  Drift Check   │  │ Model Registry │        │
│                    │  (PSI)         │  │  (MLflow +     │         │
│                    └────────┬───────┘  │   joblib)      │         │
│                             │          └───────┬───────┘         │
│                    ┌────────▼────────┐        │                 │
│                    │  Prediction    │◀───────┘                 │
│                    │  Pipeline      │                          │
│                    └────────┬───────┘                          │
│                             │                                  │
│                    ┌────────▼────────┐                          │
│                    │  Forecasts     │                          │
│                    │  (Parquet)     │                          │
│                    └────────────────┘                          │
│                                                                 │
└────────────────────────────────────────────────────────────────────┘
```

---

## Biweekly Cron Schedule

### Crontab Entry

```bash
# Biweekly retraining: Monday and Thursday 02:00 UTC
# (Runs every 2 weeks, offset to avoid weekend API load)
0 2 * * 1,4 /opt/hydro-forecast/venv/bin/python -m hydro_forecast_pt.mlops.retrain \
    --config /opt/hydro-forecast/config/base.yaml \
    >> /var/log/hydro-forecast/retrain.log 2>&1
```

### Alternative: Systemd Timer

```ini
# /etc/systemd/system/hydro-forecast-retrain.timer
[Unit]
Description=Biweekly Hydro-Forecast Retraining
Requires=hydro-forecast-retrain.service

[Timer]
OnCalendar=Mon,Thu *-*-* 02:00:00
Persistent=true
RandomizedDelaySec=30m

[Install]
WantedBy=timers.target
```

```ini
# /etc/systemd/system/hydro-forecast-retrain.service
[Unit]
Description=Hydro-Forecast Retraining Job

[Service]
Type=oneshot
User=hydro-forecast
WorkingDirectory=/opt/hydro-forecast
ExecStart=/opt/hydro-forecast/venv/bin/python -m hydro_forecast_pt.mlops.retrain \
    --config /opt/hydro-forecast/config/base.yaml
Environment=PYTHONPATH=/opt/hydro-forecast/src
StandardOutput=journal
StandardError=journal
```

---

## Environment Setup

### Required Directories

```bash
mkdir -p /opt/hydro-forecast/{config,data/{raw,processed,features,predictions},models/registry,logs}
chown -R hydro-forecast:hydro-forecast /opt/hydro-forecast
```

### Python Environment

```bash
# Using Poetry (recommended)
cd /opt/hydro-forecast
poetry install --only=main --no-dev

# Or using pip
pip install -e . --no-dev
```

### Environment Variables

```bash
# /etc/hydro-forecast/env
export HYDRO_FORECAST_CONFIG=/opt/hydro-forecast/config/base.yaml
export HYDRO_FORECAST_DATA_DIR=/opt/hydro-forecast/data
export HYDRO_FORECAST_MODEL_DIR=/opt/hydro-forecast/models/registry
export HYDRO_FORECAST_LOG_DIR=/var/log/hydro-forecast
export MLFLOW_TRACKING_URI=sqlite:////opt/hydro-forecast/mlflow.db
export OPENMETEO_BASE_URL=https://archive-api.open-meteo.com/v1/era5-land
```

---

## Model Registry

### Structure

```
models/registry/
├── random_forest/
│   ├── champion/
│   │   ├── model.joblib           # sklearn model
│   │   ├── metadata.json          # metrics, params, commit hash
│   │   └── feature_names.json     # training feature order
│   ├── challengers/
│   │   ├── v1_model.joblib
│   │   ├── v1_metadata.json
│   │   └── ...
│   └── versions.json              # version index
├── lightgbm/
│   ├── champion/
│   │   ├── model.txt              # LightGBM native format
│   │   ├── metadata.json
│   │   └── feature_names.json
│   ├── challengers/
│   └── versions.json
└── registry.db                    # SQLite index (or MLflow)
```

### Metadata Schema (`metadata.json`)

```json
{
  "model_name": "random_forest",
  "version": "v20261026_010913",
  "commit_hash": "a1b2c3d4e5f6...",
  "training_date": "2026-10-26T01:09:13",
  "data_range": {"start": "2024-01-01", "end": "2026-10-25"},
  "hyperparameters": {
    "n_estimators": 500,
    "max_depth": 15,
    "min_samples_leaf": 5,
    "max_features": 0.5
  },
  "metrics": {
    "skill_score": 0.807,
    "nse": 0.957,
    "nrmse": 0.059,
    "pic": 0.82,
    "mpiw": 0.15
  },
  "per_horizon": {...},
  "per_basin": {...},
  "is_champion": true,
  "drift_status": "OK",
  "drift_alerts": []
}
```

---

## Retraining Pipeline (`src/hydro_forecast_pt/mlops/retrain.py`)

### Main Entry Point

```bash
python -m hydro_forecast_pt.mlops.retrain \
    --config config/base.yaml \
    --run-date 2026-10-26 \
    --force-retrain false \
    --skip-drift-check false
```

### Pipeline Steps

```python
def run_retraining(run_date: datetime = None) -> Dict:
    """Complete biweekly retraining pipeline."""
    
    # 1. DATA INGESTION
    logger.info("Starting data ingestion...")
    pipeline = DataPipeline(config, data_dir)
    data = pipeline.run(
        start_date=config.start_date,
        end_date=run_date - timedelta(weeks=1)  # Last complete week
    )
    
    # 2. FEATURE ENGINEERING
    logger.info("Building features...")
    X, y = build_features_and_targets(data, config)
    
    # 3. CROSS-VALIDATION TRAINING
    logger.info("Training models with rolling origin CV...")
    cv = RollingOriginCV(**config.cv)
    
    results = {}
    for model_name in ["random_forest", "lightgbm", "climatology", 
                       "persistence", "seasonal_persistence"]:
        model = create_model(model_name, config)
        cv_results = cross_validate(model, X, y, cv)
        results[model_name] = cv_results
    
    # 4. CHAMPION/CHALLENGER EVALUATION
    logger.info("Evaluating champions...")
    champions = evaluate_champions(results, config.champion)
    
    # 5. DRIFT CHECK
    if not config.skip_drift_check:
        logger.info("Checking drift...")
        drift_results = check_drift(X, config.drift)
    
    # 6. MODEL REGISTRY UPDATE
    logger.info("Updating model registry...")
    registry = ModelRegistry(config.model_dir)
    for name, model_result in results.items():
        registry.register(
            model=model_result.best_model,
            model_name=name,
            metrics=model_result.metrics,
            commit_hash=get_git_hash(),
            is_champion=champions.get(name, False)
        )
    
    # 7. SAVE ARTIFACTS
    save_training_run(run_date, results, champions, drift_results)
    
    return {
        "status": "completed",
        "champions": champions,
        "drift": drift_results,
        "artifacts": "training_runs/{timestamp}/"
    }
```

---

## Prediction Pipeline (`src/hydro_forecast_pt/mlops/predict.py`)

### Inference Script

```bash
python -m hydro_forecast_pt.mlops.predict \
    --model random_forest \
    --output-dir data/predictions/ \
    --forecast-date 2026-10-26 \
    --horizons 1 2 3 4 5 6
```

### Pipeline Steps

```python
def generate_forecasts(model_name: str, forecast_date: date, 
                       horizons: List[int] = [1,2,3,4,5,6]) -> Dict:
    """Generate forecasts for all basins."""
    
    # 1. Load champion model
    registry = ModelRegistry(config.model_dir)
    model, metadata = registry.get_champion(model_name)
    
    # 2. Prepare latest features
    feature_date = forecast_date - timedelta(weeks=1)  # Last known data
    X_latest = prepare_features_for_date(feature_date, horizons)
    
    # 3. Generate predictions with quantiles
    forecasts = model.predict_quantiles(X_latest, quantiles=[0.1, 0.5, 0.9])
    
    # 4. Format output
    output = {
        "forecast_date": forecast_date.isoformat(),
        "model": model_name,
        "model_version": metadata["version"],
        "horizons": horizons,
        "quantiles": [0.1, 0.5, 0.9],
        "basins": ["rhine", "rhone", "inn", "ticino"],
        "predictions": {}
    }
    
    for i, basin in enumerate(output["basins"]):
        for h_idx, h in enumerate(horizons):
            for q in [0.1, 0.5, 0.9]:
                key = f"{basin}_h{h}w_q{int(q*100)}"
                output["predictions"][key] = float(forecasts[q][0, h_idx])
    
    # 5. Save
    output_path = f"data/predictions/{forecast_date.isoformat()}_{model_name}.parquet"
    pd.DataFrame([output]).to_parquet(output_path)
    
    return output
```

---

## Drift Monitoring (`src/hydro_forecast_pt/mlops/drift.py`)

### Drift Check Script

```bash
python -m hydro_forecast_pt.mlops.drift_check \
    --config config/base.yaml \
    --reference-weeks 52 \
    --current-weeks 26 \
    --threshold 0.2
```

### Alert Integration

```python
def send_drift_alert(alerts: List[str], severity: str = "warning"):
    """Send drift alerts via configured channels."""
    
    # Slack
    if config.slack_webhook:
        requests.post(config.slack_webhook, json={
            "text": f"🌊 Hydro-Forecast Drift Alert ({severity})",
            "blocks": format_alerts(alerts)
        })
    
    # Email
    if config.email_recipients:
        send_email(
            to=config.email_recipients,
            subject=f"Hydro-Forecast Drift Alert - {datetime.now().date()}",
            body=format_alerts_email(alerts)
        )
    
    # Log
    logger.warning(f"Drift alerts: {alerts}")
```

---

## Rollback Procedure

### When to Rollback

1. Champion model shows sudden performance degradation
2. Drift check reveals major covariate shift (PSI > 0.5)
3. New champion fails validation in shadow mode

### Rollback Steps

```bash
# 1. Identify previous champion
python -m hydro_forecast_pt.mlops.registry list-challengers random_forest --top-k 3

# 2. Promote previous champion
python -m hydro_forecast_pt.mlops.registry promote \
    --model random_forest \
    --version v20261011_010913 \
    --reason "Rollback due to drift in temperature_c (PSI=0.59)"

# 3. Verify
python -m hydro_forecast_pt.mlops.predict --model random_forest --verify

# 4. Alert team
send_alert("Rollback completed: random_forest v20261011_010913 promoted to champion")
```

### Automated Rollback (Optional)

```python
def auto_rollback_if_needed(model_name: str, 
                            performance_threshold: float = 0.05,
                            drift_threshold: float = 0.5):
    """Automatically rollback if performance drops significantly."""
    
    # Check recent predictions vs actuals
    recent_metrics = evaluate_recent_predictions(model_name, weeks=4)
    
    if recent_metrics["skill_score"] < performance_threshold:
        challengers = registry.list_challengers(model_name, top_k=3)
        if challengers:
            previous = challengers[0]  # Most recent challenger
            registry.promote(model_name, previous["version"], 
                           reason=f"Auto-rollback: skill dropped to {recent_metrics['skill_score']:.3f}")
            logger.warning(f"Auto-rollback triggered for {model_name}")
```

---

## Monitoring & Observability

### Key Metrics to Monitor

| Metric | Source | Alert Threshold |
|--------|--------|-----------------|
| Retraining duration | Pipeline logs | > 30 minutes |
| Champion skill score | Registry | < 0.5 |
| Drift alerts count | Drift check | > 5 features |
| Prediction latency | Prediction pipeline | > 10 seconds |
| Data freshness | Ingestion pipeline | > 48 hours old |
| API error rates | Data clients | > 5% |

### Logging

```python
# Structured JSON logging
import structlog

logger = structlog.get_logger()

logger.info("retrain_started", run_date=run_date.isoformat())
logger.info("model_trained", model="random_forest", fold=1, nse=0.95)
logger.info("champion_selected", model="random_forest", skill=0.807)
logger.warning("drift_detected", feature="temperature_c", psi=0.234)
logger.info("retrain_completed", duration_seconds=1423, champions=["rf", "lgbm"])
```

### Log Rotation

```bash
# /etc/logrotate.d/hydro-forecast
/var/log/hydro-forecast/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 hydro-forecast hydro-forecast
}
```

---

## Free Hosting Options

### GitHub Actions (Free Tier)

```yaml
# .github/workflows/retrain.yml
name: Biweekly Retraining

on:
  schedule:
    - cron: '0 2 * * 1,4'  # Mon/Thu 02:00 UTC
  workflow_dispatch:

jobs:
  retrain:
    runs-on: ubuntu-latest
    timeout-minutes: 60
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          
      - name: Install Poetry
        run: pip install poetry
        
      - name: Install dependencies
        run: poetry install --only=main
        
      - name: Run retraining
        run: poetry run python -m hydro_forecast_pt.mlops.retrain
        env:
          OPENMETEO_BASE_URL: https://archive-api.open-meteo.com/v1/era5-land
          
      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: training-run-${{ github.run_id }}
          path: training_runs/
          retention-days: 90
          
      - name: Commit model registry
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add models/registry/
          git commit -m "Update model registry [skip ci]" || true
          git push
```

### Render.com / Railway.app (Free Tier)

- Deploy as background worker
- Use cron job add-on for scheduling
- Persistent disk for model registry

### Local / VPS (Always Free)

- Cron + systemd on any Linux box
- SQLite for MLflow tracking
- No external dependencies

---

## Security Considerations

1. **API Keys**: None required (open-meteo is free, no key)
2. **Network**: Only outbound HTTPS to open-meteo.com
3. **Data**: No PII, only meteorological/hydrological time series
4. **Model Artifacts**: Stored locally, no external transmission
5. **Permissions**: Run as dedicated `hydro-forecast` user (non-root)

---

## Disaster Recovery

### Backup Strategy

```bash
# Daily backup of model registry
0 3 * * * tar -czf /backups/hydro-forecast-$(date +\%Y\%m\%d).tar.gz \
    /opt/hydro-forecast/models/registry \
    /opt/hydro-forecast/config

# Weekly backup of training runs
0 4 * * 0 tar -czf /backups/hydro-forecast-runs-weekly-$(date +\%Y\%W).tar.gz \
    /opt/hydro-forecast/training_runs
```

### Recovery Procedure

```bash
# 1. Restore config and registry
tar -xzf /backups/hydro-forecast-20261025.tar.gz -C /

# 2. Verify
python -m hydro_forecast_pt.mlops.registry list-champions

# 3. Resume cron
systemctl restart hydro-forecast-retrain.timer
```