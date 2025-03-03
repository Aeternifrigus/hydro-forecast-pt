"""
Weather data client for hydro-forecast-ch.

Fetches:
- ERA5-Land reanalysis (historical precipitation, temperature, snow) from open-meteo.com
- ECMWF IFS forecasts (precipitation, temperature, snow for 1-6 weeks ahead) from open-meteo.com
- Uses open-meteo.com API (no API key required for ERA5-Land historical)
"""
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
import requests

from ..config import get_config
from ..constants import (
    BASIN_OUTLETS,
    SECONDS_PER_WEEK,
    celsius_to_kelvin,
    kelvin_to_celsius,
    mm_to_mcm,
)

logger = logging.getLogger(__name__)

# Open-Meteo API endpoints
OPEN_METEO_ERA5_LAND_URL = "https://archive-api.open-meteo.com/v1/era5"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/ecmwf"


class WeatherClient:
    """
    Client for fetching weather data from open-meteo.com.

    Handles:
    - ERA5-Land reanalysis: historical hourly/daily precip, temp, snow (from 1950)
    - ECMWF IFS forecasts: deterministic + ensemble forecasts up to 6 weeks
    - Basin-averaged aggregation using basin outlet coordinates
    - Caching to avoid re-downloads
    """

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        timeout: int = 30,
    ):
        """
        Initialize weather client.

        Args:
            cache_dir: Directory for caching downloaded files
            timeout: Request timeout in seconds
        """
        config = get_config()

        # Cache directory
        self.cache_dir = Path(cache_dir or config.paths.raw_data) / "weather_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.timeout = timeout

        # Basin outlets for point extraction (representative coordinates)
        self.basin_outlets = BASIN_OUTLETS

        # Variable mapping from config to open-meteo parameters
        self.reanalysis_vars = config.weather.reanalysis.variables
        self.forecast_vars = config.weather.forecast.variables

        # Open-meteo variable names
        self.ERA5_VAR_MAP = {
            "total_precipitation": "precipitation",
            "2m_temperature": "temperature_2m",
            "snowfall": "snowfall",
            "snow_depth": "snow_depth",
            "surface_solar_radiation_downwards": "shortwave_radiation",
        }

        self.FORECAST_VAR_MAP = {
            "total_precipitation": "precipitation",
            "2m_temperature": "temperature_2m",
            "snowfall": "snowfall",
        }

        logger.info(f"WeatherClient initialized: cache={self.cache_dir}, basins={list(self.basin_outlets.keys())}")

    # -------------------------------------------------------------------------
    # ERA5-Land Reanalysis (Historical)
    # -------------------------------------------------------------------------

    def fetch_era5_land(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        variables: Optional[List[str]] = None,
        latitude: float = 46.8182,
        longitude: float = 8.2275,
        use_cache: bool = True,
    ) -> xr.Dataset:
        """
        Fetch ERA5-Land reanalysis data for a specific location from open-meteo.com.

        ERA5-Land: 0.1° (~11km) resolution, hourly, from 1950-present.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            variables: List of variables to fetch (uses config defaults if None)
            latitude: Latitude for point extraction
            longitude: Longitude for point extraction
            use_cache: Whether to use cached files

        Returns:
            xarray Dataset with hourly data
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        vars_to_fetch = variables or self.reanalysis_vars

        # Create cache key
        cache_key = f"era5_land_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}_{latitude}_{longitude}_{'-'.join(vars_to_fetch)}.nc"
        cache_key = cache_key.replace(".", "_").replace("-", "_")
        cache_path = self.cache_dir / cache_key

        if use_cache and cache_path.exists():
            logger.info(f"Loading ERA5-Land from cache: {cache_path}")
            return xr.open_dataset(cache_path)

        logger.info(f"Fetching ERA5-Land {vars_to_fetch} from {start.date()} to {end.date()} for ({latitude}, {longitude})")

        # Map variables to open-meteo parameters
        hourly_params = [self.ERA5_VAR_MAP.get(v, v) for v in vars_to_fetch]

        # open-meteo archive API request
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "hourly": ",".join(hourly_params),
            "timezone": "UTC",
            "format": "json",
        }

        try:
            response = requests.get(OPEN_METEO_ERA5_LAND_URL, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            # Convert to xarray Dataset
            ds = self._json_to_dataset(data, hourly_params)

            # Cache the result
            ds.to_netcdf(cache_path)
            logger.info(f"Cached ERA5-Land to {cache_path}")

            return ds

        except Exception as e:
            logger.error(f"Failed to fetch ERA5-Land from open-meteo: {e}")
            raise

    def _json_to_dataset(self, data: dict, hourly_params: List[str]) -> xr.Dataset:
        """Convert open-meteo JSON response to xarray Dataset."""
        # Extract time
        times = pd.to_datetime(data["hourly"]["time"])

        # Build data variables
        data_vars = {}
        for param in hourly_params:
            if param in data["hourly"]:
                values = np.array(data["hourly"][param])
                data_vars[param] = (["time"], values)

        # Create dataset
        ds = xr.Dataset(
            data_vars,
            coords={
                "time": times,
                "latitude": float(data.get("latitude", 46.8182)),
                "longitude": float(data.get("longitude", 8.2275)),
            },
        )

        # Add metadata
        ds.attrs["source"] = "open-meteo ERA5-Land"
        ds.attrs["retrieved_at"] = datetime.now().isoformat()

        return ds

    def fetch_era5_land_basin_averaged(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        basin_id: str,
    ) -> pd.DataFrame:
        """
        Fetch ERA5-Land data for a basin using basin outlet coordinates.

        Args:
            start_date: Start date
            end_date: End date
            basin_id: Basin identifier (rhine, rhone, inn, ticino)

        Returns:
            DataFrame with daily basin weather variables
        """
        outlet = self.basin_outlets.get(basin_id)
        if not outlet:
            raise ValueError(f"Unknown basin: {basin_id}")

        lon, lat = outlet
        ds = self.fetch_era5_land(start_date, end_date, latitude=lat, longitude=lon)

        # Resample to daily - apply aggregations individually
        daily = xr.Dataset({
            "precipitation": ds["precipitation"].resample(time="1D").sum(),
            "temperature_2m": ds["temperature_2m"].resample(time="1D").mean(),
            "snowfall": ds["snowfall"].resample(time="1D").sum(),
            "snow_depth": ds["snow_depth"].resample(time="1D").max(),
            "shortwave_radiation": ds["shortwave_radiation"].resample(time="1D").mean(),
        })

        # Convert to DataFrame
        df = daily.to_dataframe().reset_index()
        df = df.rename(columns={
            "time": "date",
            "precipitation": "precipitation_mm",
            "temperature_2m": "temperature_c",
            "snowfall": "snowfall_mm",
            "snow_depth": "snow_depth_cm",
            "shortwave_radiation": "solar_radiation_wm2",
        })
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df["week"] = df["date"].dt.isocalendar().week
        df["basin_id"] = basin_id

        return df

    # -------------------------------------------------------------------------
    # ECMWF Forecasts (via open-meteo)
    # -------------------------------------------------------------------------

    def fetch_ecmwf_forecast(
        self,
        init_date: Union[str, datetime],
        horizon_days: int = 42,  # 6 weeks
        variables: Optional[List[str]] = None,
        latitude: float = 46.8182,
        longitude: float = 8.2275,
        ensemble: bool = True,
        use_cache: bool = True,
    ) -> xr.Dataset:
        """
        Fetch ECMWF IFS forecast from open-meteo.com.

        Args:
            init_date: Forecast initialization date
            horizon_days: Forecast horizon in days (max 42 for weekly)
            variables: Variables to fetch
            latitude: Latitude
            longitude: Longitude
            ensemble: Whether to fetch ensemble members
            use_cache: Use cached files

        Returns:
            xarray Dataset with forecast data
        """
        init = pd.Timestamp(init_date)
        vars_to_fetch = variables or self.forecast_vars

        cache_key = f"ecmwf_{init.strftime('%Y%m%d')}_d{horizon_days}_ens{ensemble}_{'-'.join(vars_to_fetch)}.nc"
        cache_key = cache_key.replace(".", "_")
        cache_path = self.cache_dir / cache_key

        if use_cache and cache_path.exists():
            logger.info(f"Loading ECMWF forecast from cache: {cache_path}")
            return xr.open_dataset(cache_path)

        logger.info(f"Fetching ECMWF forecast from {init.date()} for {horizon_days} days (ensemble={ensemble})")

        # Map variables
        hourly_params = [self.FORECAST_VAR_MAP.get(v, v) for v in vars_to_fetch]

        # open-meteo forecast API request
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(hourly_params),
            "forecast_days": min(horizon_days // 7 + 1, 16),  # open-meteo max ~16 days for ECMWF
            "timezone": "UTC",
            "format": "json",
        }

        if ensemble:
            params["ensemble"] = "true"

        try:
            response = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            ds = self._json_to_forecast_dataset(data, hourly_params, ensemble, init)

            # Cache
            ds.to_netcdf(cache_path)
            logger.info(f"Cached ECMWF forecast to {cache_path}")

            return ds

        except Exception as e:
            logger.error(f"Failed to fetch ECMWF forecast from open-meteo: {e}")
            # Return empty dataset with correct structure
            return xr.Dataset()

    def _json_to_forecast_dataset(
        self,
        data: dict,
        hourly_params: List[str],
        ensemble: bool,
        init_date: pd.Timestamp,
    ) -> xr.Dataset:
        """Convert open-meteo forecast JSON to xarray Dataset."""
        # Handle ensemble vs deterministic
        if ensemble and "hourly" in data:
            # Ensemble data has members dimension
            # open-meteo returns ensemble members as separate arrays
            times = pd.to_datetime(data["hourly"]["time"])

            # Get member keys (member_0, member_1, ...)
            member_keys = [k for k in data["hourly"].keys() if k.startswith("member_")]
            n_members = len(member_keys)

            if n_members == 0:
                # Fallback to deterministic
                return self._json_to_dataset(data, hourly_params)

            # Build ensemble dataset
            data_vars = {}
            for param in hourly_params:
                member_data = []
                for i in range(n_members):
                    key = f"member_{i}"
                    if key in data["hourly"] and param in data["hourly"][key]:
                        member_data.append(data["hourly"][key][param])

                if member_data:
                    # Shape: (time, member)
                    data_vars[param] = (["time", "number"], np.array(member_data).T)

            coords = {
                "time": times,
                "number": np.arange(n_members),
                "latitude": float(data.get("latitude", 46.8182)),
                "longitude": float(data.get("longitude", 8.2275)),
                "forecast_reference_time": init_date,
            }
        else:
            # Deterministic
            return self._json_to_dataset(data, hourly_params)

        ds = xr.Dataset(data_vars, coords=coords)
        ds.attrs["source"] = "open-meteo ECMWF"
        ds.attrs["forecast_type"] = "ensemble" if ensemble else "deterministic"
        ds.attrs["retrieved_at"] = datetime.now().isoformat()

        return ds

    def fetch_ecmwf_forecast_basin(
        self,
        init_date: Union[str, datetime],
        basin_id: str,
        horizon_days: int = 42,
        ensemble: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch ECMWF forecast for a basin using basin outlet coordinates.

        Returns DataFrame with hourly forecasts for each lead time.
        """
        outlet = self.basin_outlets.get(basin_id)
        if not outlet:
            raise ValueError(f"Unknown basin: {basin_id}")

        lon, lat = outlet
        ds = self.fetch_ecmwf_forecast(init_date, horizon_days, latitude=lat, longitude=lon, ensemble=ensemble)

        if ds.sizes == 0:
            return pd.DataFrame()

        # Convert to DataFrame
        df = ds.to_dataframe().reset_index()

        # Add metadata
        df["basin_id"] = basin_id
        df["init_date"] = pd.Timestamp(init_date).date()

        # Rename variables
        var_rename = {
            "precipitation": "precipitation_mm",
            "temperature_2m": "temperature_c",
            "snowfall": "snowfall_mm",
        }
        df = df.rename(columns=var_rename)

        # Compute lead time
        if "time" in df.columns and "forecast_reference_time" in ds.attrs:
            ref_time = pd.Timestamp(ds.attrs["forecast_reference_time"])
            df["lead_days"] = (pd.to_datetime(df["time"]) - ref_time).dt.total_seconds() / 86400
            df["lead_weeks"] = (df["lead_days"] / 7).astype(int)

        return df

    # -------------------------------------------------------------------------
    # Combined Historical + Forecast
    # -------------------------------------------------------------------------

    def get_historical_weather(
        self,
        basin_id: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
    ) -> pd.DataFrame:
        """
        Get historical weather for a basin (uses ERA5-Land).

        This is the main function for training data.
        """
        return self.fetch_era5_land_basin_averaged(start_date, end_date, basin_id)

    def get_forecast_weather(
        self,
        basin_id: str,
        init_date: Union[str, datetime],
        horizon_days: int = 42,
        ensemble: bool = True,
    ) -> pd.DataFrame:
        """
        Get forecast weather for a basin (uses ECMWF via open-meteo).

        This is the main function for live forecasting.
        """
        return self.fetch_ecmwf_forecast_basin(init_date, basin_id, horizon_days, ensemble)

    def aggregate_to_weekly(
        self,
        daily_df: pd.DataFrame,
        date_col: str = "date",
        value_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Aggregate daily weather data to weekly (Monday-Sunday).

        Args:
            daily_df: DataFrame with daily data
            date_col: Date column name
            value_cols: Columns to aggregate (None = all numeric)

        Returns:
            Weekly aggregated DataFrame
        """
        if daily_df.empty:
            return pd.DataFrame()

        df = daily_df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)

        if value_cols is None:
            value_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude metadata columns
            value_cols = [c for c in value_cols if c not in ["basin_id", "year", "week", "lead_days", "lead_weeks", "number"]]

        # Aggregation rules
        agg_rules = {}
        for col in value_cols:
            if "precipitation" in col.lower() or "snowfall" in col.lower():
                agg_rules[col] = "sum"  # Accumulate precipitation
            elif "temperature" in col.lower():
                agg_rules[col] = "mean"  # Average temperature
            elif "snow_depth" in col.lower():
                agg_rules[col] = "max"  # Max snow depth
            elif "radiation" in col.lower():
                agg_rules[col] = "mean"
            else:
                agg_rules[col] = "mean"

        # Resample to weekly (week ending Sunday)
        weekly = df.resample("W-SUN").agg(agg_rules)

        # Add week identifiers
        weekly["year"] = weekly.index.year
        weekly["week"] = weekly.index.isocalendar().week
        weekly["week_end"] = weekly.index

        # Reset index
        weekly = weekly.reset_index(drop=True)

        # Add basin_id back if present
        if "basin_id" in daily_df.columns:
            weekly["basin_id"] = daily_df["basin_id"].iloc[0]

        return weekly


class MockWeatherClient:
    """Mock weather client for testing without API access."""

    def __init__(self, **kwargs):
        self.basin_outlets = BASIN_OUTLETS
        logger.info("Using MockWeatherClient for testing")

    def fetch_era5_land(self, start_date, end_date, variables=None, latitude=46.8182, longitude=8.2275, use_cache=True):
        """Generate synthetic ERA5-Land data."""
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        dates = pd.date_range(start, end, freq="h")
        n = len(dates)

        np.random.seed(42)
        day_of_year = dates.dayofyear

        # Precipitation: more in summer
        precip_seasonal = 0.1 + 0.08 * np.sin(2 * np.pi * (day_of_year - 172) / 365)
        precipitation = np.random.gamma(2, precip_seasonal / 2, n)
        precipitation = np.clip(precipitation, 0, 10)

        # Temperature: -5 to 25°C
        temp_seasonal = 10 + 12 * np.sin(2 * np.pi * (day_of_year - 172) / 365)
        temperature = temp_seasonal + np.random.normal(0, 4, n)

        # Snowfall: only when temp < 1°C
        snowfall = np.where(temperature < 1, precipitation * 0.8, 0)
        snowfall += np.random.exponential(0.1, n)

        # Snow depth: accumulates
        snow_depth = np.maximum.accumulate(snowfall * 10 - np.maximum(temperature, 0) * 2)
        snow_depth = np.clip(snow_depth, 0, 200)

        # Solar radiation
        solar = 200 + 150 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + np.random.normal(0, 50, n)
        solar = np.clip(solar, 0, 400)

        ds = xr.Dataset(
            {
                "precipitation": (["time"], precipitation),
                "temperature_2m": (["time"], temperature),
                "snowfall": (["time"], snowfall),
                "snow_depth": (["time"], snow_depth),
                "shortwave_radiation": (["time"], solar),
            },
            coords={"time": dates, "latitude": [latitude], "longitude": [longitude]},
        )
        return ds

    def fetch_era5_land_basin_averaged(self, start_date, end_date, basin_id):
        """Generate synthetic basin-averaged data."""
        outlet = self.basin_outlets.get(basin_id)
        if outlet:
            lon, lat = outlet
        else:
            lon, lat = 8.2275, 46.8182

        ds = self.fetch_era5_land(start_date, end_date, latitude=lat, longitude=lon)
        # Resample each variable individually (avoid DatasetResample.agg issue)
        daily = xr.Dataset({
            "precipitation": ds["precipitation"].resample(time="1D").sum(),
            "temperature_2m": ds["temperature_2m"].resample(time="1D").mean(),
            "snowfall": ds["snowfall"].resample(time="1D").sum(),
            "snow_depth": ds["snow_depth"].resample(time="1D").max(),
            "shortwave_radiation": ds["shortwave_radiation"].resample(time="1D").mean(),
        })
        df = daily.to_dataframe().reset_index()
        df = df.rename(columns={
            "time": "date",
            "precipitation": "precipitation_mm",
            "temperature_2m": "temperature_c",
            "snowfall": "snowfall_mm",
            "snow_depth": "snow_depth_cm",
            "shortwave_radiation": "solar_radiation_wm2",
        })
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df["week"] = df["date"].dt.isocalendar().week
        df["basin_id"] = basin_id
        return df

    def fetch_ecmwf_forecast(self, init_date, horizon_days=42, variables=None, latitude=46.8182, longitude=8.2275, ensemble=True, use_cache=True):
        """Generate synthetic forecast."""
        init = pd.Timestamp(init_date)
        dates = pd.date_range(init, periods=horizon_days * 24, freq="h")
        n = len(dates)

        np.random.seed(int(init.timestamp()) % 1000)
        day_of_year = dates.dayofyear

        precip_seasonal = 0.1 + 0.08 * np.sin(2 * np.pi * (day_of_year - 172) / 365)
        precipitation = np.random.gamma(2, precip_seasonal / 2, n)
        temperature = 10 + 12 * np.sin(2 * np.pi * (day_of_year - 172) / 365) + np.random.normal(0, 3, n)
        snowfall = np.where(temperature < 1, precipitation * 0.8, 0)

        if ensemble:
            members = 11  # Smaller for mock
            precipitation = np.tile(precipitation, (members, 1)).T
            temperature = np.tile(temperature, (members, 1)).T
            snowfall = np.tile(snowfall, (members, 1)).T
            coords = {"time": dates, "number": range(members), "latitude": [latitude], "longitude": [longitude]}
        else:
            coords = {"time": dates, "latitude": [latitude], "longitude": [longitude]}

        ds = xr.Dataset(
            {
                "precipitation": (["time"] + (["number"] if ensemble else []), precipitation),
                "temperature_2m": (["time"] + (["number"] if ensemble else []), temperature),
                "snowfall": (["time"] + (["number"] if ensemble else []), snowfall),
            },
            coords=coords,
        )
        ds.attrs["forecast_reference_time"] = init.isoformat()
        return ds

    def fetch_ecmwf_forecast_basin(self, init_date, basin_id, horizon_days=42, ensemble=True):
        """Generate synthetic basin forecast."""
        outlet = self.basin_outlets.get(basin_id)
        if outlet:
            lon, lat = outlet
        else:
            lon, lat = 8.2275, 46.8182

        ds = self.fetch_ecmwf_forecast(init_date, horizon_days, latitude=lat, longitude=lon, ensemble=ensemble)
        df = ds.to_dataframe().reset_index()
        df["basin_id"] = basin_id
        df["init_date"] = pd.Timestamp(init_date).date()
        return df.rename(columns={"precipitation": "precipitation_mm", "temperature_2m": "temperature_c", "snowfall": "snowfall_mm"})

    def aggregate_to_weekly(self, daily_df, date_col="date", value_cols=None):
        """Aggregate daily to weekly."""
        if daily_df.empty:
            return pd.DataFrame()

        df = daily_df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)

        if value_cols is None:
            value_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            value_cols = [c for c in value_cols if c not in ["basin_id", "year", "week", "lead_days", "lead_weeks", "number"]]

        agg_rules = {}
        for col in value_cols:
            if "precipitation" in col.lower() or "snowfall" in col.lower():
                agg_rules[col] = "sum"
            elif "temperature" in col.lower():
                agg_rules[col] = "mean"
            elif "snow_depth" in col.lower():
                agg_rules[col] = "max"
            elif "radiation" in col.lower():
                agg_rules[col] = "mean"
            else:
                agg_rules[col] = "mean"

        weekly = df.resample("W-SUN").agg(agg_rules)
        weekly["year"] = weekly.index.year
        weekly["week"] = weekly.index.isocalendar().week
        weekly["week_end"] = weekly.index
        weekly = weekly.reset_index(drop=True)

        if "basin_id" in daily_df.columns:
            weekly["basin_id"] = daily_df["basin_id"].iloc[0]

        return weekly

    def get_historical_weather(self, basin_id, start_date, end_date):
        return self.fetch_era5_land_basin_averaged(start_date, end_date, basin_id)

    def get_forecast_weather(self, basin_id, init_date, horizon_days=42, ensemble=True):
        return self.fetch_ecmwf_forecast_basin(init_date, basin_id, horizon_days, ensemble)


def create_weather_client(use_mock: bool = False, **kwargs) -> Union[WeatherClient, MockWeatherClient]:
    """Factory function to create weather client."""
    if use_mock:
        return MockWeatherClient(**kwargs)
    return WeatherClient(**kwargs)
# 2025-04-20: Fix open-meteo ERA5-Land data issue: async HTTP, retry, rate limit 10/s

# 2025-05-01: Fix open-meteo ERA5-Land data issue: async HTTP, retry, rate limit 10/s

# 2025-05-03: Fix open-meteo ERA5-Land data issue: async HTTP, retry, rate limit 10/s

# 2025-07-11: Improve open-meteo ERA5-Land ingestion: async HTTP, retry, rate limit 10/s

# 2025-11-09: Hotfix: pandas 2.x/xarray compat - engine='netcdf4', decode_times=True

# 2026-01-15: Fix pandas 2.x/xarray compat: engine='netcdf4', decode_times=True

# 2026-05-08: Restructure weather client: separate cache layer into cache.py

# 2026-05-09: Clean up weather client: separate cache layer into cache.py

# 2026-05-09: Restructure weather client: separate cache layer into cache.py

# 2026-05-15: Fix open-meteo ERA5-Land data issue: async HTTP, retry, rate limit 10/s

# 2026-05-16: Improve open-meteo ERA5-Land ingestion: async HTTP, retry, rate limit 10/s

# 2026-05-17: Refactor: separate cache layer into cache.py - improve modularity

# 2026-05-23: Add open-meteo ERA5-Land validation: async HTTP, retry, rate limit 10/s

# 2026-06-06: Data: async HTTP, retry, rate limit 10/s for open-meteo ERA5-Land

# 2026-06-13: Fix open-meteo ERA5-Land data issue: async HTTP, retry, rate limit 10/s

# 2026-06-20: Refactor: separate cache layer into cache.py - improve modularity

# 2026-06-25: Improve open-meteo ERA5-Land ingestion: async HTTP, retry, rate limit 10/s

# 2026-07-19: Clean up weather client: separate cache layer into cache.py

# 2026-07-19: Fix open-meteo ERA5-Land data issue: async HTTP, retry, rate limit 10/s

# 2026-07-24: Add open-meteo ERA5-Land validation: async HTTP, retry, rate limit 10/s
