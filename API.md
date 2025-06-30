# API Reference

Complete interface documentation for all data clients, model classes, and pipeline components.

---

## Data Clients

### WeatherClient (`src/hydro_forecast_pt/data/weather_client.py`)

**Purpose**: Fetch ERA5-Land reanalysis data from open-meteo.com

```python
class WeatherClient:
    def __init__(self, cache_dir: Path = Path("data/raw/weather"), rate_limit: float = 10.0)
    
    async def fetch_basin_weather(
        self,
        basin_name: str,
        start_date: date,
        end_date: date,
        variables: List[str] = None
    ) -> pd.DataFrame:
        """
        Fetch hourly weather for a basin and aggregate to weekly.
        
        Returns DataFrame with columns:
        - basin_id, year, week
        - temperature_c (mean)
        - precipitation_mm (sum)
        - radiation_mj_m2 (sum)
        - snow_depth_cm (max)
        - soil_moisture_proxy (mean)
        - derived: cdd_week, hdd_week, pet_mm (Hargreaves-Samani)
        """
    
    def get_basin_centroid(self, basin_name: str) -> Tuple[float, float]:
        """Return (lat, lon) centroid for ERA5-Land grid mapping."""
```

**Configuration** (`config/base.yaml`):
```yaml
weather:
  source: "open-meteo"
  era5_land_resolution: 0.1  # degrees
  variables: ["temperature_2m", "precipitation", "surface_solar_radiation_downwards", 
              "snow_depth", "soil_moisture_0_to_7cm"]
  cache_ttl_hours: 24
```

**Rate Limits**: 10 requests/second (respectful of open-meteo free tier)

---

### SMARDClient (`src/hydro_forecast_pt/data/smard_client.py`)

**Purpose**: Fetch German hydro generation from Federal Network Agency (SMARD) — used as proxy for Swiss RoR

```python
class SMARDClient:
    def __init__(self, cache_dir: Path = Path("data/raw/smard"))
    
    def fetch_generation(
        self,
        start_date: date,
        end_date: date,
        module_ids: List[int] = [4007, 4008, 4009],  # RoR, reservoir, pumped
        region: str = "DE"
    ) -> pd.DataFrame:
        """
        Returns quarter-hourly generation (MW) with columns:
        - timestamp, module_id, module_name, generation_mw
        """
    
    def fetch_capacity(self, module_ids: List[int] = [4007, 4008, 4009]) -> pd.DataFrame:
        """Returns installed capacity (MW) per module."""
    
    def weekly_capacity_factor(self, generation_df: pd.DataFrame, capacity_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate to weekly CF per module."""
```

**Note**: Historical weekly endpoints return 404. Uses `MockSMARDClient` fallback with realistic German RoR patterns.

---

### WASTAClient (`src/hydro_forecast_pt/data/wasta_client.py`)

**Purpose**: Swiss hydro plant registry from admin.ch OGC API

```python
class WASTAClient:
    def __init__(self, cache_dir: Path = Path("data/raw/wasta"))
    
    def fetch_plants(self) -> gpd.GeoDataFrame:
        """
        Returns GeoDataFrame with columns:
        - plant_id, name, type (run_of_river/reservoir/pumped)
        - capacity_mw, lat, lon, commissioning_year
        - basin_id (assigned via point-in-polygon)
        """
```

**Note**: OGC API endpoint returns 404. Uses `MockWASTAClient` with 5 realistic plants per basin.

---

### BFEClient (`src/hydro_forecast_pt/data/bfe_client.py`)

**Purpose**: BFE weekly reservoir fill levels and production by type

```python
class BFEClient:
    def __init__(self, cache_dir: Path = Path("data/raw/bfe"))
    
    def fetch_reservoir_fill(self, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Weekly reservoir fill levels (%) by region.
        Columns: date, region, fill_pct
        """
    
    def fetch_production(self, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Weekly production (GWh) by type.
        Columns: date, type (ror/reservoir/pumped), production_gwh
        """
```

**Note**: Excel download URLs return 404. Uses `MockBFEClient` with seasonal cycles.

---

### Mock Clients (`src/hydro_forecast_pt/data/mock_clients.py`)

Fallback implementations used when real APIs return 404:

```python
class MockSMARDClient:
    """Generates synthetic German RoR CF with seasonal pattern + noise."""
    
class MockWASTAClient:
    """Returns 5 plants/basin with realistic Swiss coordinates and capacities."""
    
class MockBFEClient:
    """Generates seasonal reservoir fill (winter low ~30%, summer high ~85%)
       and production by type with realistic proportions."""
```

All mock clients implement the same interface as real clients for seamless substitution.

---

## Feature Engineering

### HydroFeatures (`src/hydro_forecast_pt/features/hydro.py`)

```python
class HydroFeatures:
    def __init__(self, awi_alpha: float = 0.85, bucket_tau_fast: int = 2, 
                 bucket_tau_slow: int = 8, ddf: float = 3.5)
    
    def compute_awi(self, precip: pd.Series) -> pd.Series:
        """Antecedent Wetness Index: AWI_t = α·AWI_{t-1} + (1-α)·P_t"""
    
    def compute_bucket_model(self, precip: pd.Series, temp: pd.Series) -> Dict[str, pd.Series]:
        """Two-layer bucket: fast (surface) + slow (baseflow) reservoirs.
        Returns: {'fast_storage', 'slow_storage', 'fast_runoff', 'slow_runoff', 'total_runoff'}"""
    
    def compute_snowmelt(self, temp_mean: pd.Series, temp_min: pd.Series, 
                         precip: pd.Series, snow_depth: pd.Series) -> pd.Series:
        """Degree-day snowmelt with refreezing."""
    
    def compute_degree_days(self, temp_mean: pd.Series, base: float = 15.0) -> Tuple[pd.Series, pd.Series]:
        """Returns (CDD, HDD) with base temperature."""
```

---

### RollingFeatures (`src/hydro_forecast_pt/features/rolling.py`)

```python
class RollingFeatures:
    def __init__(self, windows: List[int] = [1, 2, 3, 4, 8, 12, 26, 52])
    
    def compute_precip_rolling(self, precip: pd.Series) -> pd.DataFrame:
        """Rolling sum over windows."""
    
    def compute_temp_rolling(self, temp: pd.Series) -> pd.DataFrame:
        """Rolling mean, std, min, max over windows."""
    
    def compute_expanding(self, series: pd.Series, water_year_start: int = 10) -> pd.DataFrame:
        """Expanding mean/std since water year start (Oct 1)."""
```

---

### FeaturePipeline (`src/hydro_forecast_pt/features/pipeline.py`)

```python
def build_feature_pipeline(numeric_features: List[str], 
                           categorical_features: List[str],
                           cyclic_features: List[str]) -> ColumnTransformer:
    """
    Returns sklearn ColumnTransformer:
    - StandardScaler for numeric
    - OneHotEncoder for categorical (basin_id)
    - Passthrough for cyclic (sin/cos already normalized)
    """
```

---

## Models

### RandomForestModel (`src/hydro_forecast_pt/models/random_forest.py`)

```python
class RandomForestModel(MultiHorizonModel):
    def __init__(self, horizons: List[int] = [1,2,3,4,5,6],
                 multi_model: bool = True,
                 use_quantile: bool = True,
                 **rf_params)
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "RandomForestModel":
        """X: (n_samples, n_features), y: (n_samples, n_horizons) with columns target_cf_h{h}w"""
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Returns (n_samples, n_horizons) point predictions (median)."""
    
    def predict_quantiles(self, X: pd.DataFrame, 
                          quantiles: List[float] = [0.1, 0.5, 0.9]) -> Dict[float, np.ndarray]:
        """Returns dict {quantile: (n_samples, n_horizons)}."""
    
    def get_feature_importance(self, horizon: int = None, top_n: int = 20) -> pd.DataFrame:
        """Returns DataFrame with columns: feature, importance."""
    
    def tune_hyperparameters(self, X: pd.DataFrame, y: pd.DataFrame,
                             param_grid: dict = None, n_iter: int = 30,
                             cv_splits: int = 3, gap: int = 4) -> Dict:
        """RandomizedSearchCV with TimeSeriesSplit."""
```

**Key Methods**:
- `_preprocess_features(X)`: Drops datetime cols, label-encodes basin_id
- `_create_rf(quantile)`: Creates RF or QuantileRF instance
- `_fit_single_horizon()` / `_fit_multi_output()`: Training strategies

---

### LightGBMModel (`src/hydro_forecast_pt/models/lightgbm.py`)

```python
class LightGBMModel(MultiHorizonModel):
    def __init__(self, horizons: List[int] = [1,2,3,4,5,6],
                 quantiles: List[float] = [0.1, 0.5, 0.9],
                 categorical_features: List[str] = ["basin_id"],
                 **lgbm_params)
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "LightGBMModel":
        """Native categorical handling, early stopping on validation fold."""
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Returns median predictions (quantile=0.5)."""
    
    def predict_quantiles(self, X: pd.DataFrame) -> Dict[float, np.ndarray]:
        """Returns all quantiles from single training."""
```

---

### Baselines (`src/hydro_forecast_pt/models/baselines.py`)

```python
class ClimatologyModel:
    """Weekly mean CF per basin per calendar week from training data."""
    
class PersistenceModel:
    """Last observed CF carried forward (naive)."""
    
class SeasonalPersistenceModel:
    """Same week last year (seasonal naive)."""
```

---

## Evaluation

### RollingOriginCV (`src/hydro_forecast_pt/evaluation/cv.py`)

```python
class RollingOriginCV:
    def __init__(self, initial_train: int = 104, step: int = 2,
                 test: int = 2, gap: int = 1, group_col: str = "basin_id")
    
    def split(self, X: pd.DataFrame, y: pd.DataFrame) -> Generator:
        """Yields (train_idx, test_idx) for each fold with group-aware splits."""
```

**GroupTimeSeriesSplit**: Custom implementation ensuring all samples from a basin in a fold are either train OR test (no basin appears in both).

---

### Metrics (`src/hydro_forecast_pt/evaluation/metrics.py`)

```python
def nash_sutcliffe_efficiency(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NSE = 1 - Σ(y-ŷ)²/Σ(y-ȳ)²"""

def kling_gupta_efficiency(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """KGE = 1 - √[(r-1)² + (β-1)² + (γ-1)²] where β=μ_ŷ/μ_y, γ=σ_ŷ/σ_y"""

def normalized_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """nRMSE = RMSE / (max(y) - min(y))"""

def skill_score(y_true: np.ndarray, y_pred: np.ndarray, y_clim: np.ndarray) -> float:
    """Skill = 1 - MSE_model / MSE_climatology"""

def prediction_interval_coverage(y_true: np.ndarray, q_low: np.ndarray, 
                                  q_high: np.ndarray, alpha: float = 0.1) -> float:
    """PIC = % y_true ∈ [q_low, q_high] for (1-α) interval"""

def interval_width(q_low: np.ndarray, q_high: np.ndarray) -> float:
    """Mean interval width."""
```

---

## MLOps

### RetrainPipeline (`src/hydro_forecast_pt/mlops/retrain.py`)

```python
class RetrainPipeline:
    def __init__(self, config: Dict, model_registry: ModelRegistry)
    
    def run(self, run_date: datetime = None) -> Dict:
        """
        Full biweekly retraining:
        1. Ingest new data
        2. Build features
        3. Rolling origin CV training
        4. Champion/challenger evaluation
        5. Drift check
        6. Promote/retire models
        7. Save artifacts
        """
```

### ModelRegistry (`src/hydro_forecast_pt/mlops/registry.py`)

```python
class ModelRegistry:
    def __init__(self, registry_dir: Path = Path("models/registry"))
    
    def register(self, model: Any, model_name: str, metrics: Dict, 
                 commit_hash: str, is_champion: bool) -> str:
        """Save model + metadata, return model_id."""
    
    def get_champion(self, model_name: str) -> Tuple[Any, Dict]:
        """Load current champion model + metadata."""
    
    def list_challengers(self, model_name: str, top_k: int = 3) -> List[Dict]:
        """Get top-k challengers for rollback."""
```

### DriftDetector (`src/hydro_forecast_pt/mlops/drift.py`)

```python
class PSIDriftDetector:
    def __init__(self, reference_data: pd.DataFrame, features: List[str],
                 bins: int = 10, threshold: float = 0.2)
    
    def compute_psi(self, current_data: pd.DataFrame) -> Dict[str, float]:
        """Returns PSI per feature. PSI = Σ(pᵢ - qᵢ)ln(pᵢ/qᵢ)"""
    
    def check_drift(self, current_data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Returns (has_drift, alert_messages)."""
```

---

## Pipeline Orchestration

### DataPipeline (`src/hydro_forecast_pt/data/pipeline.py`)

```python
class DataPipeline:
    def __init__(self, config: Dict, cache_dir: Path)
    
    def run(self, start_date: date, end_date: date) -> Dict[str, pd.DataFrame]:
        """
        DAG execution:
        1. weather_client.fetch_basin_weather() → weekly weather
        2. hydro_client.fetch_*() → hydro data
        3. feature_engineering.build_features() → 217 features
        4. target_construction.build_targets() → CF targets h=1..6
        Returns: {'features': X, 'targets': y, 'metadata': {...}}
        """
    
    def save_checkpoint(self, stage: str, data: Any) -> None:
        """Save intermediate results for resume capability."""
    
    def load_checkpoint(self, stage: str) -> Any:
        """Load from checkpoint."""
```

---

## Configuration Schema (`src/hydro_forecast_pt/config/schema.py`)

```python
class ModelConfig(BaseModel):
    # Random Forest
    n_estimators: int = 500
    max_depth: Optional[int] = 15
    min_samples_leaf: int = 5
    max_features: Union[str, float] = 0.5
    min_samples_split: int = 2
    bootstrap: bool = True
    oob_score: bool = True
    
    # LightGBM
    num_leaves: int = 63
    learning_rate: float = 0.03
    n_estimators: int = 1000
    min_child_samples: int = 50
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.5
    reg_lambda: float = 1.0
    
    # Common
    random_state: int = 42
    n_jobs: int = -1

class CVConfig(BaseModel):
    initial_train_weeks: int = 104
    step_weeks: int = 2
    test_weeks: int = 2
    gap_weeks: int = 1
    group_col: str = "basin_id"

class DriftConfig(BaseModel):
    reference_weeks: int = 52
    current_weeks: int = 26
    bins: int = 10
    threshold: float = 0.2
    features: List[str] = [...]  # 20 features

class ChampionConfig(BaseModel):
    skill_threshold: float = 0.01
    min_nse: float = 0.5
    max_nrmse: float = 0.3
```

---

## Error Codes

| Code | Source | Meaning |
|------|--------|---------|
| `WEATHER_429` | WeatherClient | Rate limited - exponential backoff applied |
| `WEATHER_5XX` | WeatherClient | Server error - retry with backoff |
| `SMARD_404` | SMARDClient | Historical weekly endpoint unavailable - using mock |
| `WASTA_404` | WASTAClient | OGC API unavailable - using mock |
| `BFE_404` | BFEClient | Excel URLs unavailable - using mock |
| `CV_INSUFFICIENT_DATA` | RollingOriginCV | < 3 folds available - increase initial_train |
| `LGBM_EARLY_STOP` | LightGBMModel | Early stopping triggered - check train size |
| `PSI_THRESHOLD` | PSIDriftDetector | Feature drift detected above threshold |

---

## Usage Examples

### Train a New Model
```python
from hydro_forecast_pt.data.pipeline import DataPipeline
from hydro_forecast_pt.models.random_forest import RandomForestModel
from hydro_forecast_pt.evaluation.cv import RollingOriginCV

# 1. Build features
pipeline = DataPipeline(config, Path("data"))
data = pipeline.run(date(2024, 1, 1), date(2026, 10, 25))

# 2. Train with CV
cv = RollingOriginCV(initial_train=104, step=2, test=2, gap=1)
model = RandomForestModel(horizons=[1,2,3,4,5,6])

for train_idx, test_idx in cv.split(data['features'], data['targets']):
    model.fit(data['features'].iloc[train_idx], data['targets'].iloc[train_idx])
    preds = model.predict(data['features'].iloc[test_idx])
    # evaluate...
```

### Load Champion and Predict
```python
from hydro_forecast_pt.mlops.registry import ModelRegistry

registry = ModelRegistry(Path("models/registry"))
model, metadata = registry.get_champion("random_forest")

# Prepare latest features
X_latest = prepare_features_for_date(datetime.now())
forecasts = model.predict_quantiles(X_latest)  # {0.1: ..., 0.5: ..., 0.9: ...}
```

### Check Drift
```python
from hydro_forecast_pt.mlops.drift import PSIDriftDetector

detector = PSIDriftDetector(reference_data, drift_features)
has_drift, alerts = detector.check_drift(current_data)
if has_drift:
    for alert in alerts:
        logger.warning(f"DRIFT: {alert}")
```