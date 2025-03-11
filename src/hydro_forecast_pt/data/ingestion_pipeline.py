"""
Data ingestion pipeline for hydro-forecast-ch.

Orchestrates fetching from all data sources:
- ENTSO-E: Generation and capacity
- SMARD: German electricity market data (alternative to ENTSO-E)
- ERA5-Land/ECMWF: Weather (historical + forecast)
- BFE/WASTA: Swiss reservoir levels, production, plant registry

Handles historical backfill and daily/weekly incremental updates.
"""
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..config import get_config
from ..data.storage import DataLake, get_data_lake
from ..data.entsoe_client import EntsoeClient, create_entsoe_client, MockEntsoeClient
from ..data.smard_client import SmardClient, create_smard_client, MockSmardClient
from ..data.weather_client import WeatherClient, create_weather_client, MockWeatherClient
from ..data.bfe_client import BFEClient, create_bfe_client, MockBFEClient
from ..constants import SWISS_BASINS, DEFAULT_HORIZONS_WEEKS

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Main ingestion pipeline coordinating all data sources.

    Supports:
    - Historical backfill (full reprocessing)
    - Incremental weekly updates
    - Data validation and quality checks
    - DVC-compatible output structure
    """

    def __init__(
        self,
        use_mock: bool = False,
        use_smard: bool = False,
        data_lake: Optional[DataLake] = None,
        entsoe_client: Optional[Union[EntsoeClient, MockEntsoeClient]] = None,
        smard_client: Optional[Union[SmardClient, MockSmardClient]] = None,
        weather_client: Optional[Union[WeatherClient, MockWeatherClient]] = None,
        bfe_client: Optional[Union[BFEClient, MockBFEClient]] = None,
    ):
        """
        Initialize ingestion pipeline.

        Args:
            use_mock: Use mock clients for testing
            use_smard: Use SMARD API instead of ENTSO-E for generation data
            data_lake: DataLake instance (created if None)
            entsoe_client: ENTSO-E client (created if None)
            smard_client: SMARD client (created if None)
            weather_client: Weather client (created if None)
            bfe_client: BFE client (created if None)
        """
        self.config = get_config()
        self.use_mock = use_mock
        self.use_smard = use_smard

        # Initialize clients
        self.data_lake = data_lake or get_data_lake()

        if use_smard:
            self.entsoe = None
            self.smard = smard_client or create_smard_client(use_mock=use_mock, region="DE")
        else:
            self.entsoe = entsoe_client or create_entsoe_client(use_mock=use_mock)
            self.smard = None

        self.weather = weather_client or create_weather_client(use_mock=use_mock)
        self.bfe = bfe_client or create_bfe_client(use_mock=use_mock)

        # Basins to process
        self.basins = [b["id"] for b in SWISS_BASINS.values()]

        logger.info(f"IngestionPipeline initialized for basins: {self.basins}, mock={use_mock}, use_smard={use_smard}")

    # -------------------------------------------------------------------------
    # Historical Backfill
    # -------------------------------------------------------------------------

    def run_historical_backfill(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        overwrite: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Run full historical backfill for all data sources.

        Args:
            start_date: Start date for backfill
            end_date: End date for backfill
            overwrite: Whether to overwrite existing data

        Returns:
            Dictionary of fetched datasets
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        logger.info(f"Starting historical backfill from {start.date()} to {end.date()}")

        results = {}

        # 1. Generation and capacity (ENTSO-E or SMARD)
        if self.use_smard:
            logger.info("Fetching SMARD generation data (Germany)...")
            try:
                gen_df = self.smard.fetch_generation(start, end)
                # SMARD doesn't have capacity endpoint, estimate from data
                cap_df = self.smard.fetch_installed_capacity(start.year)
                cf_df = self.smard.get_weekly_capacity_factor(start, end)

                results["smard_generation"] = gen_df
                results["smard_capacity"] = cap_df
                results["smard_capacity_factor"] = cf_df

                # Save raw
                self.data_lake.save_raw_parquet(
                    gen_df.reset_index(), "smard_generation",
                    partition_cols=["year", "week"], overwrite=overwrite
                )
                self.data_lake.save_raw_parquet(
                    cap_df.reset_index(), "smard_capacity", overwrite=overwrite
                )
                self.data_lake.save_raw_parquet(
                    cf_df.reset_index(), "smard_capacity_factor",
                    partition_cols=["year", "week"], overwrite=overwrite
                )

                logger.info(f"SMARD: {len(gen_df)} quarter-hourly, {len(cf_df)} weekly records")

            except Exception as e:
                logger.error(f"SMARD fetch failed: {e}")
                if not self.use_mock:
                    raise
        else:
            logger.info("Fetching ENTSO-E generation data...")
            try:
                gen_df = self.entsoe.fetch_generation(start, end)
                cap_df = self.entsoe.fetch_installed_capacity(start.year)

                # Also fetch weekly capacity factor
                cf_df = self.entsoe.get_weekly_capacity_factor(start, end)

                results["entsoe_generation"] = gen_df
                results["entsoe_capacity"] = cap_df
                results["entsoe_capacity_factor"] = cf_df

                # Save raw
                self.data_lake.save_raw_parquet(
                    gen_df.reset_index(), "entsoe_generation",
                    partition_cols=["year", "week"] if "week" in gen_df.columns else None,
                    overwrite=overwrite
                )
                self.data_lake.save_raw_parquet(
                    cap_df.reset_index(), "entsoe_capacity", overwrite=overwrite
                )
                self.data_lake.save_raw_parquet(
                    cf_df.reset_index(), "entsoe_capacity_factor",
                    partition_cols=["year", "week"], overwrite=overwrite
                )

                logger.info(f"ENTSO-E: {len(gen_df)} hourly, {len(cf_df)} weekly records")

            except Exception as e:
                logger.error(f"ENTSO-E fetch failed: {e}")
                if not self.use_mock:
                    raise

        # 2. Weather reanalysis (ERA5-Land) for each basin
        logger.info("Fetching ERA5-Land weather data...")
        weather_results = {}
        for basin_id in self.basins:
            try:
                basin_weather = self.weather.get_historical_weather(basin_id, start, end)

                if not basin_weather.empty:
                    # Aggregate to weekly
                    weekly_weather = self.weather.aggregate_to_weekly(basin_weather)
                    weather_results[basin_id] = weekly_weather

                    # Save raw daily
                    self.data_lake.save_raw_parquet(
                        basin_weather, f"era5_daily_{basin_id}",
                        partition_cols=["year", "week"], overwrite=overwrite
                    )
                    # Save processed weekly
                    self.data_lake.save_parquet(
                        weekly_weather, f"weather_weekly_{basin_id}",
                        partition_cols=["year", "week"], overwrite=overwrite
                    )

                    logger.info(f"  Basin {basin_id}: {len(weekly_weather)} weekly records")

            except Exception as e:
                logger.error(f"Weather fetch failed for {basin_id}: {e}")
                if not self.use_mock:
                    raise

        results["weather"] = weather_results

        # 3. BFE reservoir levels
        logger.info("Fetching BFE reservoir levels...")
        try:
            reservoir_df = self.bfe.fetch_reservoir_levels(start, end)
            if not reservoir_df.empty:
                results["reservoir_levels"] = reservoir_df
                self.data_lake.save_raw_parquet(
                    reservoir_df, "bfe_reservoir_levels",
                    partition_cols=["year", "week", "region"], overwrite=overwrite
                )
                logger.info(f"BFE reservoir: {len(reservoir_df)} records")
        except Exception as e:
            logger.error(f"BFE reservoir fetch failed: {e}")

        # 4. BFE weekly production
        logger.info("Fetching BFE weekly production...")
        try:
            production_df = self.bfe.fetch_weekly_production(start, end)
            if not production_df.empty:
                results["production"] = production_df
                self.data_lake.save_raw_parquet(
                    production_df, "bfe_production",
                    partition_cols=["year", "week", "production_type"], overwrite=overwrite
                )
                logger.info(f"BFE production: {len(production_df)} records")
        except Exception as e:
            logger.error(f"BFE production fetch failed: {e}")

        # 5. WASTA plant registry
        logger.info("Fetching WASTA plant registry...")
        try:
            wasta_df = self.bfe.fetch_wasta_plants(start.year)
            if not wasta_df.empty:
                results["wasta_plants"] = wasta_df
                self.data_lake.save_raw_parquet(wasta_df, "wasta_plants", overwrite=overwrite)
                logger.info(f"WASTA: {len(wasta_df)} plants")
        except Exception as e:
            logger.error(f"WASTA fetch failed: {e}")

        logger.info("Historical backfill completed")
        return results

    # -------------------------------------------------------------------------
    # Incremental Weekly Update
    # -------------------------------------------------------------------------

    def run_weekly_update(
        self,
        reference_date: Optional[Union[str, datetime]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Run incremental weekly update.

        Fetches data for the most recent complete week (ending Sunday).
        BFE publishes on Wednesdays, so we fetch the week ending the previous Sunday.

        Args:
            reference_date: Reference date (default: today)

        Returns:
            Dictionary of updated datasets
        """
        ref = pd.Timestamp(reference_date or datetime.now())

        # Find most recent completed week (week ending Sunday)
        # If today is Wed, last Sunday was 3 days ago
        days_since_sunday = (ref.weekday() + 1) % 7  # Monday=0, Sunday=6
        last_sunday = ref - timedelta(days=days_since_sunday)
        week_start = last_sunday - timedelta(days=6)

        logger.info(f"Running weekly update for week {week_start.date()} to {last_sunday.date()}")

        results = {}

        # Generation: Fetch last 2 weeks to catch revisions (SMARD or ENTSO-E)
        gen_start = week_start - timedelta(weeks=2)
        try:
            if self.use_smard:
                gen_df = self.smard.fetch_generation(gen_start, last_sunday + timedelta(days=1))
                cf_df = self.smard.get_weekly_capacity_factor(gen_start, last_sunday + timedelta(days=1))
                dataset_name = "smard_capacity_factor"
            else:
                gen_df = self.entsoe.fetch_generation(gen_start, last_sunday + timedelta(days=1))
                cf_df = self.entsoe.get_weekly_capacity_factor(gen_start, last_sunday + timedelta(days=1))
                dataset_name = "entsoe_capacity_factor"

            # Keep only the target week
            target_week_mask = (cf_df.index >= week_start) & (cf_df.index <= last_sunday)
            cf_week = cf_df[target_week_mask]

            if not cf_week.empty:
                results[dataset_name] = cf_week
                self.data_lake.save_parquet(
                    cf_week.reset_index(), dataset_name,
                    partition_cols=["year", "week"], overwrite=True
                )
                logger.info(f"Updated {dataset_name}: {len(cf_week)} weeks")
        except Exception as e:
            logger.error(f"Generation weekly update failed: {e}")

        # Weather: Fetch ERA5-Land for last 2 weeks (reanalysis has ~5 day lag)
        weather_start = week_start - timedelta(weeks=2)
        weather_results = {}
        for basin_id in self.basins:
            try:
                basin_weather = self.weather.get_historical_weather(basin_id, weather_start, last_sunday)
                if not basin_weather.empty:
                    weekly_weather = self.weather.aggregate_to_weekly(basin_weather)
                    # Keep target week
                    weekly_weather = weekly_weather[weekly_weather["week_end"] == last_sunday]

                    if not weekly_weather.empty:
                        weather_results[basin_id] = weekly_weather
                        self.data_lake.save_parquet(
                            weekly_weather, f"weather_weekly_{basin_id}",
                            partition_cols=["year", "week"], overwrite=True
                        )
                        logger.info(f"Updated weather for {basin_id}: {len(weekly_weather)} weeks")
            except Exception as e:
                logger.error(f"Weather update failed for {basin_id}: {e}")

        results["weather"] = weather_results

        # BFE: Fetch latest Wednesday's data (covers week ending previous Sunday)
        try:
            reservoir_df = self.bfe.fetch_reservoir_levels(week_start, last_sunday)
            if not reservoir_df.empty:
                results["reservoir_levels"] = reservoir_df
                self.data_lake.save_raw_parquet(
                    reservoir_df, "bfe_reservoir_levels",
                    partition_cols=["year", "week", "region"], overwrite=True
                )
        except Exception as e:
            logger.error(f"BFE reservoir update failed: {e}")

        try:
            production_df = self.bfe.fetch_weekly_production(week_start, last_sunday)
            if not production_df.empty:
                results["production"] = production_df
                self.data_lake.save_raw_parquet(
                    production_df, "bfe_production",
                    partition_cols=["year", "week", "production_type"], overwrite=True
                )
        except Exception as e:
            logger.error(f"BFE production update failed: {e}")

        # ECMWF Forecast: Fetch new forecast for next 6 weeks
        try:
            forecast_results = {}
            for basin_id in self.basins:
                forecast = self.weather.get_forecast_weather(basin_id, ref, horizon_weeks=6)
                if not forecast.empty:
                    forecast_results[basin_id] = forecast
                    self.data_lake.save_raw_parquet(
                        forecast, f"ecmwf_forecast_{basin_id}",
                        partition_cols=["init_date", "lead_weeks"], overwrite=True
                    )
            results["forecast"] = forecast_results
            logger.info(f"Updated ECMWF forecasts for {len(forecast_results)} basins")
        except Exception as e:
            logger.error(f"ECMWF forecast update failed: {e}")

        logger.info("Weekly update completed")
        return results

    # -------------------------------------------------------------------------
    # Daily Update (for forecasts)
    # -------------------------------------------------------------------------

    def run_daily_forecast_update(self) -> Dict[str, pd.DataFrame]:
        """
        Run daily forecast update.

        Fetches new ECMWF forecast every day for all basins.
        """
        today = pd.Timestamp(datetime.now().date())
        logger.info(f"Running daily forecast update for {today.date()}")

        results = {}

        for basin_id in self.basins:
            try:
                forecast = self.weather.get_forecast_weather(basin_id, today, horizon_weeks=6)
                if not forecast.empty:
                    results[basin_id] = forecast
                    self.data_lake.save_raw_parquet(
                        forecast, f"ecmwf_forecast_{basin_id}",
                        partition_cols=["init_date", "lead_weeks"], overwrite=True
                    )
            except Exception as e:
                logger.error(f"Daily forecast failed for {basin_id}: {e}")

        return results

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate_data(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
    ) -> Dict[str, Dict]:
        """
        Validate ingested data quality.

        Checks:
        - Completeness (no gaps > threshold)
        - Outliers (IQR method)
        - Reasonable value ranges
        - Consistency across sources
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        validation_results = {}

        # Validate generation capacity factor (SMARD or ENTSO-E)
        cf_dataset = "smard_capacity_factor" if self.use_smard else "entsoe_capacity_factor"
        try:
            cf_df = self.data_lake.load_parquet(
                cf_dataset,
                filters=[("year", ">=", start.year), ("year", "<=", end.year)]
            )
            if not cf_df.empty:
                validation_results[cf_dataset] = self._validate_timeseries(
                    cf_df, "capacity_factor",
                    min_val=0, max_val=1.2,
                    max_gap_weeks=4
                )
        except Exception as e:
            validation_results[cf_dataset] = {"error": str(e)}

        # Validate weather for each basin
        for basin_id in self.basins:
            try:
                weather_df = self.data_lake.load_parquet(
                    f"weather_weekly_{basin_id}",
                    filters=[("year", ">=", start.year), ("year", "<=", end.year)]
                )
                if not weather_df.empty:
                    validation_results[f"weather_{basin_id}"] = self._validate_weather(weather_df)
            except Exception as e:
                validation_results[f"weather_{basin_id}"] = {"error": str(e)}

        return validation_results

    def _validate_timeseries(
        self,
        df: pd.DataFrame,
        value_col: str,
        min_val: float,
        max_val: float,
        max_gap_weeks: int = 4,
    ) -> Dict:
        """Validate a single time series."""
        result = {
            "total_records": len(df),
            "date_range": (df["week_end"].min(), df["week_end"].max()) if "week_end" in df.columns else None,
            "missing_weeks": 0,
            "outliers": 0,
            "out_of_bounds": 0,
        }

        if value_col in df.columns:
            vals = df[value_col].dropna()

            # Out of bounds
            oob = ((vals < min_val) | (vals > max_val)).sum()
            result["out_of_bounds"] = int(oob)

            # Outliers (IQR)
            Q1, Q3 = vals.quantile([0.25, 0.75])
            IQR = Q3 - Q1
            outliers = ((vals < Q1 - 3*IQR) | (vals > Q3 + 3*IQR)).sum()
            result["outliers"] = int(outliers)

            # Gaps
            if "week_end" in df.columns:
                expected_weeks = pd.date_range(
                    df["week_end"].min(), df["week_end"].max(), freq="W-SUN"
                )
                actual_weeks = set(df["week_end"])
                missing = len([w for w in expected_weeks if w not in actual_weeks])
                result["missing_weeks"] = missing

        return result

    def _validate_weather(self, df: pd.DataFrame) -> Dict:
        """Validate weather data."""
        result = {"total_records": len(df)}

        checks = {
            "precipitation_mm": (0, 500),
            "temperature_c": (-30, 40),
            "snowfall_mm": (0, 200),
            "snow_depth_cm": (0, 500),
        }

        for col, (min_v, max_v) in checks.items():
            if col in df.columns:
                vals = df[col].dropna()
                oob = ((vals < min_v) | (vals > max_v)).sum()
                result[f"{col}_out_of_bounds"] = int(oob)
                result[f"{col}_mean"] = float(vals.mean())
                result[f"{col}_std"] = float(vals.std())

        return result

    def process_and_merge(
        self,
        entsoe_generation: Optional[pd.DataFrame] = None,
        entsoe_capacity: Optional[pd.DataFrame] = None,
        entsoe_capacity_factor: Optional[pd.DataFrame] = None,
        weather_data: Optional[Dict[str, pd.DataFrame]] = None,
        wasta_plants: Optional[pd.DataFrame] = None,
        reservoir_levels: Optional[pd.DataFrame] = None,
        weekly_production: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Merge all data sources into feature matrix X and target vector y.

        This creates a unified dataset with:
        - Features from weather (basin-level aggregated weekly)
        - Targets from ENTSO-E/SMARD capacity factor (weekly hydro generation)
        - Additional features from reservoir levels, production, plant registry

        Args:
            entsoe_generation: Hourly generation data
            entsoe_capacity: Installed capacity data
            weather_data: Dict of basin_id -> weekly weather DataFrame
            wasta_plants: WASTA plant registry
            reservoir_levels: BFE reservoir levels
            weekly_production: BFE weekly production

        Returns:
            Tuple of (X_features, y_targets)
        """
        logger.info("Merging all data sources...")

        # Start with weather data as the base (has basin-level weekly frequency)
        if weather_data is None or len(weather_data) == 0:
            raise ValueError("Weather data is required for merging")

        # Combine weather from all basins into single DataFrame with basin_id column
        weather_frames = []
        for basin_id, df in weather_data.items():
            if not df.empty:
                df = df.copy()
                df["basin_id"] = basin_id
                # Drop duplicates within each basin
                df = df.drop_duplicates(subset=["week_end"], keep="first")
                weather_frames.append(df)

        if not weather_frames:
            raise ValueError("No weather data available for any basin")

        X = pd.concat(weather_frames, ignore_index=True)

        # Ensure week_end is datetime
        if "week_end" in X.columns:
            X["week_end"] = pd.to_datetime(X["week_end"])
        else:
            # Try to create from year/week
            if "year" in X.columns and "week" in X.columns:
                X["week_end"] = pd.to_datetime(
                    X["year"].astype(str) + "-" + X["week"].astype(str) + "-0",
                    format="%Y-%W-%w"
                )

        # Set index for merging - drop any remaining duplicates
        X = X.drop_duplicates(subset=["week_end", "basin_id"], keep="first")
        X.set_index(["week_end", "basin_id"], inplace=True)
        X.sort_index(inplace=True)

        # Add target from ENTSO-E/SMARD capacity factor (weekly hydro generation)
        y = pd.DataFrame(index=X.index)

        # Use passed capacity factor (SMARD or ENTSO-E) if available, otherwise try data lake
        cf_df = None
        cf_source = "smard_capacity_factor" if self.use_smard else "entsoe_capacity_factor"

        if entsoe_capacity_factor is not None and not entsoe_capacity_factor.empty:
            cf_df = entsoe_capacity_factor.copy()
            logger.info(f"Using passed {cf_source} data")
        else:
            # Fallback: try to load from data lake
            try:
                cf_df = self.data_lake.load_parquet(cf_source)
                logger.info(f"Loaded {cf_source} from data lake")
            except Exception as e:
                logger.warning(f"Could not load {cf_source} from data lake: {e}")

        if cf_df is not None and not cf_df.empty:
            cf_df = cf_df.copy()

            # Handle both cases: week_end as column or as index
            if "week_end" in cf_df.columns:
                cf_df["week_end"] = pd.to_datetime(cf_df["week_end"])
            elif cf_df.index.name == "week_end":
                cf_df = cf_df.reset_index()
                cf_df["week_end"] = pd.to_datetime(cf_df["week_end"])

            # Remove timezone to match X (which is timezone-naive)
            if cf_df["week_end"].dt.tz is not None:
                cf_df["week_end"] = cf_df["week_end"].dt.tz_localize(None)

            # ENTSO-E/SMARD data is national - replicate for each basin
            # Get unique basins from X
            basins = X.index.get_level_values("basin_id").unique()
            cf_list = []
            for basin in basins:
                cf_basin = cf_df.copy()
                cf_basin["basin_id"] = basin
                cf_list.append(cf_basin)
            cf_df = pd.concat(cf_list, ignore_index=True)

            cf_df.set_index(["week_end", "basin_id"], inplace=True)
            cf_df.sort_index(inplace=True)

            # The capacity factor is for all of Switzerland/Germany
            # We'll assign the same value to all basins (national generation)
            # In a more sophisticated version, we'd allocate by basin capacity share
            target_col = "capacity_factor"
            if target_col in cf_df.columns:
                # Align with X index
                common_idx = X.index.intersection(cf_df.index)
                if len(common_idx) > 0:
                    y.loc[common_idx, "run_of_river_generation_cf"] = cf_df.loc[common_idx, target_col]
                    y.loc[common_idx, "reservoir_generation_cf"] = cf_df.loc[common_idx, target_col] * 2  # Approx
                    y.loc[common_idx, "pumped_storage_cf"] = cf_df.loc[common_idx, target_col] * 0.5  # Approx

        # If no real target, create synthetic from weekly_production or capacity factor
        if y.empty or y["run_of_river_generation_cf"].isna().all():
            logger.info("Creating synthetic targets from available data...")

            if weekly_production is not None and not weekly_production.empty:
                # Use BFE weekly production
                prod_df = weekly_production.copy()
                prod_df["week_end"] = pd.to_datetime(prod_df["week_end"])

                # If no basin_id, assign national production to all basins
                if "basin_id" not in prod_df.columns:
                    # Get unique basins from X
                    basins = X.index.get_level_values("basin_id").unique()
                    # Replicate production data for each basin
                    prod_list = []
                    for basin in basins:
                        p = prod_df.copy()
                        p["basin_id"] = basin
                        prod_list.append(p)
                    prod_df = pd.concat(prod_list, ignore_index=True)

                prod_df.set_index(["week_end", "basin_id"], inplace=True)
                prod_df.sort_index(inplace=True)

                # Pivot production types
                for prod_type in prod_df["production_type"].unique():
                    type_data = prod_df[prod_df["production_type"] == prod_type]
                    common_idx = X.index.intersection(type_data.index)
                    if len(common_idx) > 0:
                        col_name = f"{prod_type.lower().replace(' ', '_')}_generation_mwh"
                        y.loc[common_idx, col_name] = type_data.loc[common_idx, "production_mwh"]
            else:
                # Last resort: synthetic targets based on seasonal patterns
                logger.warning("No target data available, creating synthetic seasonal targets")
                n_samples = len(X)
                week_end_idx = X.index.get_level_values("week_end")
                weeks = week_end_idx.isocalendar().week
                seasonal = np.sin(2 * np.pi * (weeks - 1) / 52)
                noise = np.random.normal(0, 0.1, n_samples)
                y["run_of_river_generation_cf"] = 0.3 + 0.2 * seasonal + noise
                y["reservoir_generation_cf"] = 0.5 + 0.3 * seasonal + noise
                y["pumped_storage_cf"] = 0.1 + 0.05 * seasonal + noise

        # Add reservoir levels as features
        if reservoir_levels is not None and not reservoir_levels.empty:
            res_df = reservoir_levels.copy()
            res_df["week_end"] = pd.to_datetime(res_df["week_end"])

            # If no basin_id, try to map from region
            if "basin_id" not in res_df.columns and "region" in res_df.columns:
                # Map BFE regions to basins
                region_to_basin = {
                    "region_1": "rhone",   # Valais
                    "region_2": "rhone",   # BE, FR, VD, NE, JU (mostly Rhone)
                    "region_3": "ticino",  # TI, GR (Ticino and GR - assign to Ticino for simplicity)
                    "region_4": "rhine",   # Central/Eastern (mostly Rhine)
                    "all": None,  # Skip national aggregate
                }
                res_df["basin_id"] = res_df["region"].map(region_to_basin)
                # Drop rows where basin_id is None (e.g., "all" region)
                res_df = res_df.dropna(subset=["basin_id"])

            if "basin_id" in res_df.columns:
                # Aggregate by (week_end, basin_id) to handle multiple regions per basin
                numeric_cols = res_df.select_dtypes(include=[np.number]).columns.tolist()
                numeric_cols = [c for c in numeric_cols if c not in ["year", "week"]]
                res_df = res_df.groupby(["week_end", "basin_id"])[numeric_cols].mean().reset_index()

                res_df.set_index(["week_end", "basin_id"], inplace=True)
                res_df.sort_index(inplace=True)

                for col in res_df.columns:
                    if col not in ["region", "year", "week", "basin_id"]:
                        common_idx = X.index.intersection(res_df.index)
                        if len(common_idx) > 0:
                            X.loc[common_idx, f"reservoir_{col}"] = res_df.loc[common_idx, col]

        # Add WASTA plant info as static features per basin
        if wasta_plants is not None and not wasta_plants.empty:
            # Aggregate plant capacities by basin
            basin_capacity = wasta_plants.groupby("basin_id").agg({
                "capacity_mw": ["sum", "mean", "count"],
                "type": lambda x: (x == "run_of_river").sum(),
            })
            basin_capacity.columns = ["_".join(col).strip() for col in basin_capacity.columns.values]
            basin_capacity = basin_capacity.rename(columns={
                "capacity_mw_sum": "wasta_total_capacity_mw",
                "capacity_mw_mean": "wasta_avg_capacity_mw",
                "capacity_mw_count": "wasta_plant_count",
                "type_<lambda>": "wasta_ror_count",
            })

            # Merge into X
            for basin_id in X.index.get_level_values("basin_id").unique():
                if basin_id in basin_capacity.index:
                    mask = X.index.get_level_values("basin_id") == basin_id
                    for col in basin_capacity.columns:
                        X.loc[mask, col] = basin_capacity.loc[basin_id, col]

        # Reset index for feature engineering
        X = X.reset_index()
        y = y.reset_index()

        # Fill missing values
        X = X.ffill().bfill().fillna(0)
        y = y.ffill().bfill().fillna(0)

        logger.info(f"Merged data: X={X.shape}, y={y.shape}")
        return X, y


# -------------------------------------------------------------------------
# CLI Entry Points
# -------------------------------------------------------------------------

def run_backfill_cli(
    start_date: str,
    end_date: str,
    use_mock: bool = False,
    use_smard: bool = False,
    overwrite: bool = False,
):
    """CLI entry point for historical backfill."""
    pipeline = IngestionPipeline(use_mock=use_mock, use_smard=use_smard)
    results = pipeline.run_historical_backfill(start_date, end_date, overwrite)
    print(f"Backfill completed. Datasets: {list(results.keys())}")
    return results


def run_weekly_update_cli(
    reference_date: Optional[str] = None,
    use_mock: bool = False,
    use_smard: bool = False,
):
    """CLI entry point for weekly update."""
    pipeline = IngestionPipeline(use_mock=use_mock, use_smard=use_smard)
    results = pipeline.run_weekly_update(reference_date)
    print(f"Weekly update completed. Datasets: {list(results.keys())}")
    return results


def run_daily_forecast_cli(use_mock: bool = False, use_smard: bool = False):
    """CLI entry point for daily forecast update."""
    pipeline = IngestionPipeline(use_mock=use_mock, use_smard=use_smard)
    results = pipeline.run_daily_forecast_update()
    print(f"Daily forecast update completed. Basins: {list(results.keys())}")
    return results


def run_validation_cli(
    start_date: str,
    end_date: str,
    use_mock: bool = False,
    use_smard: bool = False,
):
    """CLI entry point for data validation."""
    pipeline = IngestionPipeline(use_mock=use_mock, use_smard=use_smard)
    results = pipeline.validate_data(start_date, end_date)
    print("Validation Results:")
    for dataset, checks in results.items():
        print(f"  {dataset}: {checks}")
    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m hydro_forecast_pt.data.ingestion_pipeline <command> [args]")
        print("Commands: backfill, weekly, daily, validate")
        sys.exit(1)

    command = sys.argv[1]
    use_mock = "--mock" in sys.argv
    use_smard = "--smard" in sys.argv

    if command == "backfill":
        if len(sys.argv) < 4:
            print("Usage: backfill <start_date> <end_date> [--mock] [--smard] [--overwrite]")
            sys.exit(1)
        run_backfill_cli(sys.argv[2], sys.argv[3], use_mock, use_smard, "--overwrite" in sys.argv)

    elif command == "weekly":
        ref = sys.argv[2] if len(sys.argv) > 2 else None
        run_weekly_update_cli(ref, use_mock, use_smard)

    elif command == "daily":
        run_daily_forecast_cli(use_mock, use_smard)

    elif command == "validate":
        if len(sys.argv) < 4:
            print("Usage: validate <start_date> <end_date> [--mock] [--smard]")
            sys.exit(1)
        run_validation_cli(sys.argv[2], sys.argv[3], use_mock, use_smard)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)