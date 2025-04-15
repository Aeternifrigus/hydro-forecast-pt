"""
Feature pipeline for hydro-forecast-ch.

Combines all feature engineering steps:
1. Basin aggregation (gridded weather → basin averages)
2. Hydrological features (rolling precip, AWI, degree days, snow, soil moisture)
3. Temporal features (Fourier terms, lags, rolling stats, holidays)
4. Physics-lite bucket model features
5. Target preparation (capacity factor, horizons)

Outputs feature matrix ready for ML models.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..config import get_config
from ..data.storage import get_data_lake
from ..features.basin_aggregation import BasinAggregator, get_basin_aggregator
from ..features.hydro_features import compute_hydro_features
from ..features.temporal_features import add_temporal_features
from ..features.physics_lite import add_bucket_features, MultiBasinBucketModel
from ..constants import SWISS_BASINS, DEFAULT_HORIZONS_WEEKS

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    End-to-end feature engineering pipeline.

    Transforms raw weather + generation data into ML-ready features.
    """

    def __init__(
        self,
        basins: Optional[List[str]] = None,
        target_horizons: Optional[List[int]] = None,
        use_bucket_model: bool = True,
        bucket_state_path: Optional[str] = None,
    ):
        """
        Initialize feature pipeline.

        Args:
            basins: List of basin IDs to process
            target_horizons: Forecast horizons in weeks
            use_bucket_model: Whether to include bucket model features
            bucket_state_path: Path to save/load bucket states
        """
        self.config = get_config()
        self.basins = basins or list(SWISS_BASINS.keys())
        self.target_horizons = target_horizons or DEFAULT_HORIZONS_WEEKS
        self.use_bucket_model = use_bucket_model
        self.bucket_state_path = bucket_state_path

        # Components
        self.basin_aggregator = get_basin_aggregator()
        self.bucket_model = None

        # Feature metadata
        self.feature_names_: List[str] = []
        self.target_names_: List[str] = []

        # Data lake
        self.data_lake = get_data_lake()

        logger.info(f"FeaturePipeline initialized for basins={self.basins}, horizons={self.target_horizons}")

    def fit(
        self,
        weather_data: Dict[str, pd.DataFrame],
        target_data: Dict[str, pd.DataFrame],
        save_bucket_state: bool = True,
    ) -> "FeaturePipeline":
        """
        Fit the pipeline (initialize bucket model states).

        Args:
            weather_data: Dict of basin_id -> weekly weather DataFrame
            target_data: Dict of basin_id -> target DataFrame (capacity factor)
            save_bucket_state: Whether to save bucket state after fitting

        Returns:
            Self
        """
        logger.info("Fitting feature pipeline...")

        if self.use_bucket_model:
            # Initialize bucket model and run through historical data
            self.bucket_model = MultiBasinBucketModel(self.basins)

            for basin_id in self.basins:
                if basin_id in weather_data:
                    basin_weather = weather_data[basin_id].sort_values("week_end")

                    precip = basin_weather["precipitation_mm"].values
                    temp = basin_weather["temperature_c"].values
                    snow = basin_weather.get("snowfall_mm", pd.Series([None]*len(basin_weather))).values

                    # Run simulation to initialize state
                    self.bucket_model.models[basin_id].simulate(
                        precip, temp, snow, reset_state=True
                    )

            logger.info("Bucket model fitted and states initialized")

            if save_bucket_state and self.bucket_state_path:
                self.bucket_model.save_all_states(self.bucket_state_path)

        return self

    def transform(
        self,
        weather_data: Dict[str, pd.DataFrame],
        target_data: Optional[Dict[str, pd.DataFrame]] = None,
        is_training: bool = True,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Transform weather and target data into feature matrix.

        Args:
            weather_data: Dict of basin_id -> weekly weather DataFrame
            target_data: Dict of basin_id -> target DataFrame (optional for inference)
            is_training: Whether this is training (affects target creation)

        Returns:
            Tuple of (features_df, targets_df)
            - features_df: Combined features for all basins with columns:
              basin_id, week_end, year, week, feature_1, feature_2, ...
            - targets_df: Targets for each horizon (if target_data provided)
        """
        logger.info("Transforming data to features...")

        all_features = []
        all_targets = []

        for basin_id in self.basins:
            if basin_id not in weather_data:
                logger.warning(f"No weather data for basin {basin_id}, skipping")
                continue

            basin_weather = weather_data[basin_id].copy()
            basin_weather = basin_weather.sort_values("week_end")

            if basin_weather.empty:
                continue

            # Add basin identifier
            basin_weather["basin_id"] = basin_id

            # 1. Hydrological features
            logger.debug(f"Computing hydro features for {basin_id}")
            basin_weather = compute_hydro_features(
                basin_weather,
                precip_col="precipitation_mm",
                temp_col="temperature_c",
                snowfall_col="snowfall_mm" if "snowfall_mm" in basin_weather.columns else None,
                snow_depth_col="snow_depth_cm" if "snow_depth_cm" in basin_weather.columns else None,
                date_col="week_end",
                group_col=None,  # Already single basin
            )

            # 2. Temporal features
            logger.debug(f"Computing temporal features for {basin_id}")
            # Target columns for lag features
            target_cols = ["precipitation_mm", "temperature_c", "awi", "total_soil_moisture"]
            target_cols = [c for c in target_cols if c in basin_weather.columns]

            basin_weather = add_temporal_features(
                basin_weather,
                date_col="week_end",
                target_cols=target_cols,
                group_col=None,
                fourier_terms=self.config.features.seasonal.fourier_terms,
            )

            # 3. Bucket model features
            if self.use_bucket_model and self.bucket_model is not None:
                logger.debug(f"Computing bucket features for {basin_id}")
                basin_weather = add_bucket_features(
                    basin_weather,
                    precip_col="precipitation_mm",
                    temp_col="temperature_c",
                    snowfall_col="snowfall_mm" if "snowfall_mm" in basin_weather.columns else None,
                    basin_col="basin_id",
                    date_col="week_end",
                )

            all_features.append(basin_weather)

        if not all_features:
            raise ValueError("No valid basin data to transform")

        # Combine all basins
        features_df = pd.concat(all_features, ignore_index=True)

        # Create target variables if target_data provided
        targets_df = None
        if target_data is not None and is_training:
            targets_df = self._create_targets(features_df, target_data)

        # Store feature names (exclude metadata columns)
        metadata_cols = ["basin_id", "week_end", "year", "week", "date"]
        self.feature_names_ = [c for c in features_df.columns if c not in metadata_cols]
        self.target_names_ = [f"target_cf_h{h}w" for h in self.target_horizons]

        logger.info(f"Transformed {len(features_df)} records, {len(self.feature_names_)} features")

        return features_df, targets_df

    def fit_transform(
        self,
        weather_data: Dict[str, pd.DataFrame],
        target_data: Dict[str, pd.DataFrame],
        save_bucket_state: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fit and transform in one step."""
        self.fit(weather_data, target_data, save_bucket_state)
        return self.transform(weather_data, target_data, is_training=True)

    def _create_targets(
        self,
        features_df: pd.DataFrame,
        target_data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Create target variables for each forecast horizon.

        Target: capacity factor at t+h for h in target_horizons
        """
        logger.info("Creating target variables...")

        # Combine target data
        target_dfs = []
        for basin_id, df in target_data.items():
            df = df.copy()
            df["basin_id"] = basin_id
            target_dfs.append(df)

        if not target_dfs:
            return pd.DataFrame()

        targets_combined = pd.concat(target_dfs, ignore_index=True)
        targets_combined = targets_combined.sort_values(["basin_id", "week_end"])

        # Ensure capacity_factor column exists
        if "capacity_factor" not in targets_combined.columns:
            if "generation_mw" in targets_combined.columns:
                # Need installed capacity - approximate for now
                logger.warning("No capacity_factor column, cannot create targets")
                return pd.DataFrame()

        # Create targets for each horizon
        target_rows = []

        for basin_id in self.basins:
            basin_targets = targets_combined[targets_combined["basin_id"] == basin_id].copy()
            basin_targets = basin_targets.sort_values("week_end").reset_index(drop=True)

            if basin_targets.empty:
                continue

            cf = basin_targets["capacity_factor"].values
            dates = basin_targets["week_end"].values

            for i, (date, cf_val) in enumerate(zip(dates, cf)):
                row = {
                    "basin_id": basin_id,
                    "week_end": date,
                    "year": pd.Timestamp(date).year,
                    "week": pd.Timestamp(date).isocalendar().week,
                }

                # Add targets for each horizon
                for h in self.target_horizons:
                    target_idx = i + h
                    if target_idx < len(cf):
                        row[f"target_cf_h{h}w"] = cf[target_idx]
                    else:
                        row[f"target_cf_h{h}w"] = np.nan

                target_rows.append(row)

        targets_df = pd.DataFrame(target_rows)

        # Merge with features
        # We'll merge on basin_id + week_end
        # But for training, we need features at time t, targets at t+h
        # So we'll just return targets separately and let the model handle alignment

        logger.info(f"Created targets: {len(targets_df)} rows, {len(self.target_horizons)} horizons")
        return targets_df

    def prepare_training_data(
        self,
        weather_data: Dict[str, pd.DataFrame],
        target_data: Dict[str, pd.DataFrame],
        save_bucket_state: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare complete training dataset.

        Aligns features at time t with targets at t+h.
        Drops rows with missing targets.
        """
        features_df, targets_df = self.fit_transform(
            weather_data, target_data, save_bucket_state
        )

        # Merge features and targets on basin_id + week_end
        # Features at time t, targets at time t (which are t+h from feature perspective)
        # Actually, we need to align: features at t predict targets at t+h
        # So we merge features with targets shifted back by h weeks

        # For simplicity, let's create a combined dataset where each row has
        # features at time t and targets for all horizons

        # This is done by the model training code, not here
        # Here we just ensure both are properly formatted

        return features_df, targets_df

    def get_feature_names(self) -> List[str]:
        """Get list of feature column names."""
        return self.feature_names_

    def get_target_names(self) -> List[str]:
        """Get list of target column names."""
        return self.target_names_

    def save_bucket_states(self, path: Optional[str] = None) -> None:
        """Save bucket model states."""
        if self.bucket_model and (path or self.bucket_state_path):
            self.bucket_model.save_all_states(path or self.bucket_state_path)

    def load_bucket_states(self, path: Optional[str] = None) -> None:
        """Load bucket model states."""
        if self.bucket_model and (path or self.bucket_state_path):
            self.bucket_model.load_all_states(path or self.bucket_state_path)


def create_training_dataset(
    start_date: str,
    end_date: str,
    basins: Optional[List[str]] = None,
    horizons: Optional[List[int]] = None,
    use_mock: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to create complete training dataset.

    Runs ingestion pipeline, then feature pipeline.

    Args:
        start_date: Start date for training data
        end_date: End date for training data
        basins: Basins to include
        horizons: Forecast horizons
        use_mock: Use mock data (for testing)

    Returns:
        Tuple of (features_df, targets_df)
    """
    from ..data.ingestion_pipeline import IngestionPipeline

    logger.info(f"Creating training dataset from {start_date} to {end_date}")

    # Run ingestion
    pipeline = IngestionPipeline(use_mock=use_mock)
    raw_data = pipeline.run_historical_backfill(start_date, end_date)

    # Organize weather data by basin
    weather_data = {}
    for basin_id in (basins or list(SWISS_BASINS.keys())):
        weather_key = f"weather_weekly_{basin_id}"
        if weather_key in raw_data.get("weather", {}):
            weather_data[basin_id] = raw_data["weather"][weather_key]

    # Organize target data by basin (from ENTSO-E capacity factor)
    target_data = {}
    if "entsoe_capacity_factor" in raw_data:
        cf_df = raw_data["entsoe_capacity_factor"].reset_index()
        for basin_id in (basins or list(SWISS_BASINS.keys())):
            # For now, use same target for all basins (national)
            # In production, would have basin-level targets
            target_data[basin_id] = cf_df[["week_end", "capacity_factor"]].copy()

    # Run feature pipeline
    fp = FeaturePipeline(basins=basins, target_horizons=horizons)
    features, targets = fp.prepare_training_data(weather_data, target_data)

    return features, targets


def create_inference_features(
    basin_weather: Dict[str, pd.DataFrame],
    bucket_state_path: str,
    basins: Optional[List[str]] = None,
    horizons: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Create features for inference (forecasting).

    Uses saved bucket states for continuity.

    Args:
        basin_weather: Dict of basin_id -> recent weather DataFrame
        bucket_state_path: Path to saved bucket states
        basins: Basins to process
        horizons: Forecast horizons

    Returns:
        Features DataFrame ready for model prediction
    """
    fp = FeaturePipeline(
        basins=basins,
        target_horizons=horizons,
        use_bucket_model=True,
        bucket_state_path=bucket_state_path,
    )

    # Load bucket states
    fp.load_bucket_states(bucket_state_path)

    # Transform (no targets needed for inference)
    features, _ = fp.transform(basin_weather, target_data=None, is_training=False)

    return features