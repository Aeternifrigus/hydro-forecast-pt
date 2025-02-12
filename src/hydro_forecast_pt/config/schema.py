"""
Configuration loading for hydro-forecast-ch using Pydantic Settings.
Supports YAML config files and environment variable overrides.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EntsoeConfig(BaseModel):
    """ENTSO-E API configuration."""
    area_code: str = "CH"
    production_type: str = "B16"
    resolution: str = "PT60M"
    installed_capacity_code: str = "A68"
    revision_window_days: int = 30
    api_token: Optional[str] = None  # Set via environment variable


class WeatherReanalysisConfig(BaseModel):
    """Reanalysis weather data configuration."""
    source: str = "era5-land"
    variables: List[str] = Field(default_factory=lambda: [
        "total_precipitation", "2m_temperature", "snowfall", "snow_depth", "surface_solar_radiation_downwards"
    ])
    bbox: List[float] = Field(default_factory=lambda: [5.9, 45.8, 10.5, 47.8])
    resolution: float = 0.1
    frequency: str = "daily"


class WeatherForecastConfig(BaseModel):
    """Weather forecast configuration."""
    source: str = "ecmwf_hres"
    variables: List[str] = Field(default_factory=lambda: [
        "total_precipitation", "2m_temperature", "snowfall"
    ])
    horizon_weeks: int = 6
    ensemble_members: int = 51
    update_frequency: str = "daily"


class WeatherConfig(BaseModel):
    """Weather data configuration."""
    reanalysis: WeatherReanalysisConfig = Field(default_factory=WeatherReanalysisConfig)
    forecast: WeatherForecastConfig = Field(default_factory=WeatherForecastConfig)


class BasinConfig(BaseModel):
    """Basin configuration."""
    id: str
    name: str
    countries: List[str]
    major_sub_basins: List[str]
    gauge_stations: List[str]
    area_km2: float
    swiss_area_km2: float


class BorderPlantConfig(BaseModel):
    """Border plant configuration."""
    name: str
    basin: str
    shared_with: str
    capacity_mw: float
    type: str


class BfeWastaConfig(BaseModel):
    """Swiss Federal Office of Energy / WASTA configuration."""
    base_url: str = "https://www.bfe.admin.ch/bfe/en/home/supply/statistics/geodata.html"
    reservoir_levels_url: str = "https://www.bfe.admin.ch/bfe/en/home/supply/statistics/reservoir-levels.html"
    production_url: str = "https://www.bfe.admin.ch/bfe/en/home/supply/statistics/electricity-production.html"
    wasta_plant_list_url: str = "https://www.bfe.admin.ch/bfe/en/home/supply/statistics/wasta.html"
    update_day: str = "wednesday"
    data_format: str = "excel"
    reservoir_regions: List[str] = Field(default_factory=lambda: ["all", "region_1", "region_2", "region_3", "region_4"])


class PrecipitationFeaturesConfig(BaseModel):
    """Precipitation feature configuration."""
    rolling_windows_weeks: List[int] = Field(default_factory=lambda: [1, 2, 3, 4, 8, 12, 26])
    awi_decay_weeks: int = 8


class TemperatureFeaturesConfig(BaseModel):
    """Temperature feature configuration."""
    degree_day_base_heating: float = 15.0
    degree_day_base_cooling: float = 20.0
    degree_day_base_snowmelt: float = 0.0
    snow_melt_factor: float = 3.5


class SeasonalFeaturesConfig(BaseModel):
    """Seasonal feature configuration."""
    fourier_terms: int = 4


class PhysicsLiteConfig(BaseModel):
    """Physics-lite bucket model configuration."""
    enabled: bool = True
    bucket_type: str = "two_layer_snow"
    parameters: Dict[str, float] = Field(default_factory=lambda: {
        "fast_capacity_mm": 100,
        "slow_capacity_mm": 500,
        "snow_capacity_mm": 1000,
        "k_fast": 0.3,
        "k_slow": 0.02,
        "k_snow_melt": 0.1,
        "snow_temp_threshold": 0.0,
        "snow_melt_factor": 3.5,
    })


class FeaturesConfig(BaseModel):
    """Feature engineering configuration."""
    precipitation: PrecipitationFeaturesConfig = Field(default_factory=PrecipitationFeaturesConfig)
    temperature: TemperatureFeaturesConfig = Field(default_factory=TemperatureFeaturesConfig)
    seasonal: SeasonalFeaturesConfig = Field(default_factory=SeasonalFeaturesConfig)
    physics_lite: PhysicsLiteConfig = Field(default_factory=PhysicsLiteConfig)


class RandomForestConfig(BaseModel):
    """Random Forest model configuration."""
    n_estimators: int = 500
    max_depth: int = 20
    min_samples_split: int = 5
    min_samples_leaf: int = 10
    max_features: str = "sqrt"
    n_jobs: int = -1
    random_state: int = 42


class LightGBMConfig(BaseModel):
    """LightGBM model configuration."""
    num_leaves: int = 31
    learning_rate: float = 0.03
    n_estimators: int = 500
    min_child_samples: int = 50
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.5
    reg_lambda: float = 0.5
    objective: str = "regression"  # Use regression for point predictions
    quantiles: List[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])
    random_state: int = 42
    n_jobs: int = -1


class LSTMConfig(BaseModel):
    """LSTM model configuration."""
    lookback_weeks: int = 52
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    max_epochs: int = 100
    early_stopping_patience: int = 15


class TFTConfig(BaseModel):
    """Temporal Fusion Transformer configuration."""
    lookback_weeks: int = 52
    hidden_size: int = 128
    lstm_layers: int = 2
    attention_heads: int = 4
    dropout: float = 0.1
    batch_size: int = 32
    learning_rate: float = 0.001
    max_epochs: int = 100
    early_stopping_patience: int = 15


class ModelsConfig(BaseModel):
    """All model configurations."""
    baselines: List[str] = Field(default_factory=lambda: [
        "climatology", "persistence", "seasonal_persistence", "linear_trend"
    ])
    random_forest: RandomForestConfig = Field(default_factory=RandomForestConfig)
    lightgbm: LightGBMConfig = Field(default_factory=LightGBMConfig)
    lstm: LSTMConfig = Field(default_factory=LSTMConfig)
    tft: TFTConfig = Field(default_factory=TFTConfig)


class SplitterConfig(BaseModel):
    """Rolling origin splitter configuration."""
    type: str = "rolling_origin"
    initial_train_weeks: int = 104
    step_weeks: int = 13
    max_folds: int = 5
    gap_weeks: int = 2


class ComparisonPeriodConfig(BaseModel):
    """Comparison period for evaluation."""
    name: str
    months: List[int]


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""
    splitter: SplitterConfig = Field(default_factory=SplitterConfig)
    metrics: List[str] = Field(default_factory=lambda: [
        "nrmse", "nse", "kge", "skill_score", "picp", "mpiw", "crps"
    ])
    comparison_periods: List[ComparisonPeriodConfig] = Field(default_factory=lambda: [
        ComparisonPeriodConfig(name="wet_months", months=[10, 11, 12, 1, 2, 3]),
        ComparisonPeriodConfig(name="dry_months", months=[4, 5, 6, 7, 8, 9]),
        ComparisonPeriodConfig(name="snow_melt", months=[4, 5, 6]),
        ComparisonPeriodConfig(name="low_flow", months=[7, 8, 9, 10]),
    ])


class MLflowConfig(BaseModel):
    """MLflow configuration."""
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_prefix: str = "hydro-forecast-ch"
    artifact_location: str = "mlruns"


class DVCConfig(BaseModel):
    """DVC configuration."""
    remote: str = "s3://hydro-forecast-ch-dvc"
    cache_dir: str = ".dvc/cache"


class RetrainConfig(BaseModel):
    """Retrain configuration."""
    schedule: str = "biweekly"
    day_of_week: str = "thursday"
    champion_threshold: float = 0.01


class DriftMonitoringConfig(BaseModel):
    """Drift monitoring configuration."""
    feature_drift_threshold_ks: float = 0.1
    feature_drift_threshold_psi: float = 0.2
    prediction_drift_threshold_sigma: float = 2.0
    performance_decay_threshold: float = 0.2


class MonitoringScheduleConfig(BaseModel):
    """Monitoring schedule configuration."""
    feature_drift: str = "daily"
    prediction_drift: str = "daily"
    performance: str = "weekly"
    full_retrain_eval: str = "monthly"


class MLOpsConfig(BaseModel):
    """MLOps configuration."""
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    dvc: DVCConfig = Field(default_factory=DVCConfig)
    retrain: RetrainConfig = Field(default_factory=RetrainConfig)
    drift_monitoring: DriftMonitoringConfig = Field(default_factory=DriftMonitoringConfig)
    monitoring_schedule: MonitoringScheduleConfig = Field(default_factory=MonitoringScheduleConfig)


class DashboardConfig(BaseModel):
    """Dashboard configuration."""
    title: str = "Swiss Run-of-River Hydropower Forecast"
    update_frequency: str = "weekly"
    components: List[str] = Field(default_factory=lambda: [
        "forecast_chart", "basin_selector", "model_version",
        "baseline_comparison", "historical_performance",
        "feature_importance", "reservoir_levels"
    ])
    map_center: List[float] = Field(default_factory=lambda: [46.8, 8.2])
    map_zoom: int = 8


class PathsConfig(BaseModel):
    """File paths configuration."""
    raw_data: str = "data/raw"
    processed_data: str = "data/processed"
    features: str = "data/features"
    models: str = "models"
    metrics: str = "metrics"
    mlflow: str = "mlruns"
    dashboard: str = "dashboard"
    logs: str = "logs"


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/hydro_forecast.log"


class ProjectConfig(BaseModel):
    """Main project configuration."""
    name: str = "hydro-forecast-ch"
    country: str = "CH"
    target_type: str = "hydro_run_of_river"
    frequency: str = "weekly"
    forecast_horizons: List[int] = Field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    target_variable: str = "capacity_factor"


class Config(BaseSettings):
    """Root configuration class."""
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    entsoe: EntsoeConfig = Field(default_factory=EntsoeConfig)
    weather: WeatherConfig = Field(default_factory=WeatherConfig)
    basins: List[BasinConfig] = Field(default_factory=list)
    border_plants: List[BorderPlantConfig] = Field(default_factory=list)
    bfe_wasta: BfeWastaConfig = Field(default_factory=BfeWastaConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    mlops: MLOpsConfig = Field(default_factory=MLOpsConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    @field_validator("basins", mode="before")
    @classmethod
    def parse_basins(cls, v: Union[List[Dict], List[BasinConfig]]) -> List[BasinConfig]:
        """Parse basins from list of dicts."""
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return [BasinConfig(**b) for b in v]
        return v

    @field_validator("border_plants", mode="before")
    @classmethod
    def parse_border_plants(cls, v: Union[List[Dict], List[BorderPlantConfig]]) -> List[BorderPlantConfig]:
        """Parse border plants from list of dicts."""
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return [BorderPlantConfig(**b) for b in v]
        return v


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """
    Load configuration from YAML file and environment variables.

    Args:
        config_path: Path to YAML config file. If None, searches standard locations.

    Returns:
        Config object with all settings.
    """
    # Default config paths to search
    if config_path is None:
        search_paths = [
            Path("config/config.yaml"),
            Path("../config/config.yaml"),
            Path("../../config/config.yaml"),
            Path("/Users/kumarkaustuvdas/hydro-forecast-pt/config/config.yaml"),
        ]
        for path in search_paths:
            if path.exists():
                config_path = path
                break

    config_data = {}

    # Load from YAML if found
    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f) or {}

    # Create config object (will also load from .env via pydantic-settings)
    config = Config(**config_data)

    return config


# Global config instance (lazy-loaded)
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global config instance (singleton)."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config() -> None:
    """Reset global config (useful for testing)."""
    global _config
    _config = None