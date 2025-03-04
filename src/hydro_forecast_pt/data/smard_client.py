"""
SMARD API client for hydro-forecast-ch.

Fetches actual generation data from Germany's Federal Network Agency (SMARD).
SMARD provides German electricity market data including generation by source.
We use German run-of-river hydro as a proxy for Swiss generation patterns,
or fetch cross-border flows for Swiss-specific data.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import requests

from ..config import get_config
from ..constants import SECONDS_PER_WEEK

logger = logging.getLogger(__name__)


class SmardClient:
    """
    Client for fetching German/Swiss electricity data from SMARD API.
    
    SMARD (Strommarktdaten) is Germany's Federal Network Agency open data portal.
    Provides:
    - Actual generation per production type (quarter-hourly)
    - Cross-border flows (import/export)
    - Market prices
    
    For Swiss hydro, we can use:
    - German run-of-river hydro as proxy (similar Alpine hydrology)
    - Cross-border flows CH<->DE for Swiss net generation
    """

    BASE_URL = "https://www.smard.de/app/chart_data"
    
    # SMARD filter IDs for different data series
    FILTERS = {
        # German generation
        "de_hydro_run_of_river": "1223",      # Wasserkraft - Laufwasserkraftwerke
        "de_hydro_reservoir": "1224",          # Wasserkraft - Speicherkraftwerke
        "de_hydro_pumped": "1225",             # Wasserkraft - Pumpspeicherkraftwerke
        "de_hydro_total": "4067",              # Wasserkraft gesamt
        # Cross-border flows (Germany <-> Switzerland)
        "de_ch_import": "4169",                # Import aus Schweiz
        "de_ch_export": "4170",                # Export nach Schweiz
        # Swiss generation (if available)
        "ch_hydro_run_of_river": "5092",       # CH Laufwasserkraft
        "ch_hydro_reservoir": "5093",          # CH Speicherkraft
        "ch_hydro_total": "5091",              # CH Wasserkraft gesamt
    }

    def __init__(self, region: str = "DE", timeout: int = 30):
        """
        Initialize SMARD client.

        Args:
            region: Market region ("DE" for Germany, "CH" for Switzerland)
            timeout: Request timeout in seconds
        """
        self.region = region
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "hydro-forecast-ch/1.0 (research)"
        })
        logger.info(f"SMARD client initialized for region {region}")

    def _build_url(self, filter_id: str, timestamp: int) -> str:
        """Build SMARD API URL for a filter at a specific timestamp."""
        return f"{self.BASE_URL}/{filter_id}/DE/index_{filter_id}_{timestamp}.json"

    def _fetch_filter_data(self, filter_id: str, start: datetime, end: datetime) -> pd.DataFrame:
        """
        Fetch data for a specific filter over a date range.
        
        SMARD provides data in quarter-hourly resolution (15-min intervals).
        Data is split into weekly files. We need to fetch multiple files.
        """
        all_data = []
        current = start
        
        while current < end:
            # SMARD uses timestamps in milliseconds for file naming
            # Weekly files - get the Monday of the week
            monday = current - timedelta(days=current.weekday())
            timestamp = int(monday.timestamp() * 1000)
            
            url = self._build_url(filter_id, timestamp)
            
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                
                if "series" in data and data["series"]:
                    for series in data["series"]:
                        # series format: [timestamp_ms, value]
                        for point in series:
                            ts = pd.Timestamp(point[0], unit="ms", tz="UTC")
                            val = point[1]
                            if val is not None and not np.isnan(val):
                                all_data.append({"datetime": ts, "value": val})
                
            except requests.RequestException as e:
                logger.warning(f"Failed to fetch SMARD data for {filter_id} week {monday}: {e}")
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to parse SMARD data for {filter_id} week {monday}: {e}")
            
            current = monday + timedelta(weeks=1)
        
        if not all_data:
            return pd.DataFrame(columns=["datetime", "value"])
        
        df = pd.DataFrame(all_data)
        df = df.drop_duplicates(subset="datetime").sort_values("datetime")
        df = df.set_index("datetime")
        df.index.name = "datetime"
        
        return df

    def fetch_generation(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
        production_type: str = "hydro_run_of_river",
    ) -> pd.DataFrame:
        """
        Fetch actual generation per production type.
        
        Args:
            start: Start date (inclusive)
            end: End date (exclusive)
            production_type: One of 'hydro_run_of_river', 'hydro_reservoir', 
                           'hydro_pumped', 'hydro_total'
        
        Returns:
            DataFrame with datetime index and generation in MW (quarter-hourly)
        """
        start_dt = pd.Timestamp(start).tz_localize("UTC")
        end_dt = pd.Timestamp(end).tz_localize("UTC")
        
        filter_key = f"{self.region.lower()}_{production_type}"
        if filter_key not in self.FILTERS:
            # Try without region prefix
            filter_key = production_type
        
        if filter_key not in self.FILTERS:
            raise ValueError(f"Unknown production_type: {production_type}. Available: {list(self.FILTERS.keys())}")
        
        filter_id = self.FILTERS[filter_key]
        logger.info(f"Fetching SMARD {filter_key} ({filter_id}) for {self.region} from {start_dt} to {end_dt}")
        
        df = self._fetch_filter_data(filter_id, start_dt, end_dt)
        
        if df.empty:
            logger.warning(f"No data returned for {filter_key}")
            return pd.DataFrame(columns=["generation_mw"])
        
        df = df.rename(columns={"value": "generation_mw"})
        df["generation_mw"] = df["generation_mw"].clip(lower=0)  # No negative generation
        
        logger.info(f"Fetched {len(df)} quarter-hourly records for {filter_key}")
        return df

    def fetch_cross_border_flow(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
        direction: str = "import",  # "import" or "export"
    ) -> pd.DataFrame:
        """
        Fetch cross-border flows between Germany and Switzerland.
        
        Args:
            start: Start date
            end: End date
            direction: "import" (CH->DE) or "export" (DE->CH)
        
        Returns:
            DataFrame with flow in MW (positive = flow in direction)
        """
        filter_key = f"de_ch_{direction}"
        if filter_key not in self.FILTERS:
            raise ValueError(f"Unknown direction: {direction}")
        
        filter_id = self.FILTERS[filter_key]
        start_dt = pd.Timestamp(start).tz_localize("UTC")
        end_dt = pd.Timestamp(end).tz_localize("UTC")
        
        logger.info(f"Fetching SMARD cross-border {direction} ({filter_id}) from {start_dt} to {end_dt}")
        
        df = self._fetch_filter_data(filter_id, start_dt, end_dt)
        
        if df.empty:
            return pd.DataFrame(columns=["flow_mw"])
        
        df = df.rename(columns={"value": "flow_mw"})
        return df

    def aggregate_to_hourly(
        self,
        quarterly_df: pd.DataFrame,
        value_col: str = "generation_mw",
        agg_func: str = "mean",
    ) -> pd.DataFrame:
        """
        Aggregate quarter-hourly (15-min) data to hourly.
        
        Args:
            quarterly_df: DataFrame with datetime index and value column
            value_col: Column to aggregate
            agg_func: Aggregation function ('mean', 'sum')
        
        Returns:
            Hourly DataFrame
        """
        if quarterly_df.empty:
            return pd.DataFrame(columns=[value_col])
        
        df = quarterly_df.copy()
        
        if agg_func == "mean":
            hourly = df[value_col].resample("h").mean()
        elif agg_func == "sum":
            hourly = df[value_col].resample("h").sum()
        else:
            raise ValueError(f"Unknown agg_func: {agg_func}")
        
        hourly_df = hourly.to_frame(name=value_col)
        hourly_df.index.name = "datetime"
        return hourly_df

    def aggregate_to_weekly(
        self,
        hourly_df: pd.DataFrame,
        value_col: str = "generation_mw",
        agg_func: str = "mean",
    ) -> pd.DataFrame:
        """
        Aggregate hourly data to weekly (Monday-Sunday weeks).
        
        Args:
            hourly_df: DataFrame with datetime index and value column
            value_col: Column to aggregate
            agg_func: Aggregation function ('mean', 'sum', 'max')
        
        Returns:
            Weekly DataFrame with week ending Sunday
        """
        if hourly_df.empty:
            return pd.DataFrame(columns=[value_col])
        
        df = hourly_df.copy()
        
        if agg_func == "mean":
            weekly = df[value_col].resample("W-SUN").mean()
        elif agg_func == "sum":
            weekly = df[value_col].resample("W-SUN").sum()
        elif agg_func == "max":
            weekly = df[value_col].resample("W-SUN").max()
        else:
            raise ValueError(f"Unknown agg_func: {agg_func}")
        
        weekly_df = weekly.to_frame(name=value_col)
        weekly_df.index.name = "week_end"
        
        # Add week number and year
        weekly_df["year"] = weekly_df.index.year
        weekly_df["week"] = weekly_df.index.isocalendar().week
        
        return weekly_df

    def get_weekly_generation(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
        production_type: str = "hydro_run_of_river",
    ) -> pd.DataFrame:
        """
        Get weekly aggregated generation directly.
        
        Args:
            start: Start date
            end: End date
            production_type: Production type
        
        Returns:
            Weekly generation DataFrame
        """
        quarterly = self.fetch_generation(start, end, production_type)
        hourly = self.aggregate_to_hourly(quarterly)
        return self.aggregate_to_weekly(hourly)

    def get_weekly_capacity_factor(
        self,
        start: Union[str, datetime],
        end: Union[str, datetime],
        production_type: str = "hydro_run_of_river",
        capacity_mw: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Calculate weekly capacity factor.
        
        Args:
            start: Start date
            end: End date
            production_type: Production type
            capacity_mw: Installed capacity in MW (if known)
        
        Returns:
            DataFrame with weekly capacity factor
        """
        weekly_gen = self.get_weekly_generation(start, end, production_type)
        
        if weekly_gen.empty:
            return pd.DataFrame(columns=["capacity_factor"])
        
        # If capacity not provided, estimate from data (max generation * 1.1)
        if capacity_mw is None:
            capacity_mw = weekly_gen["generation_mw"].max() * 1.1
            logger.info(f"Estimated capacity: {capacity_mw:.0f} MW")
        
        weekly_gen["capacity_factor"] = weekly_gen["generation_mw"] / capacity_mw
        weekly_gen["capacity_factor"] = weekly_gen["capacity_factor"].clip(0, 1.2)
        
        return weekly_gen[["generation_mw", "capacity_factor", "year", "week"]]


class MockSmardClient:
    """
    Mock client for testing without network access.
    Generates synthetic German/Swiss hydro data with realistic patterns.
    """
    
    def __init__(self, region: str = "DE", capacity_mw: float = 5000, **kwargs):
        self.region = region
        self.capacity_mw = capacity_mw
        logger.info(f"Using MockSmardClient for {region} (capacity: {capacity_mw} MW)")

    def fetch_generation(self, start, end, production_type=None):
        """Generate synthetic quarter-hourly generation data."""
        start_dt = pd.Timestamp(start).tz_localize("UTC")
        end_dt = pd.Timestamp(end).tz_localize("UTC")
        
        dates = pd.date_range(start_dt, end_dt, freq="15min", inclusive="left")
        n = len(dates)
        
        # Synthetic German/Swiss hydro run-of-river pattern:
        # - Seasonal: higher in spring/summer (snowmelt + rain)
        # - Daily: diurnal pattern (higher daytime)
        # - Weekly: weekend slightly lower
        
        np.random.seed(42)
        
        # Base seasonal pattern (capacity factor 0.2-0.8)
        day_of_year = dates.dayofyear
        # Peak in June (snowmelt + rain), minimum in winter
        seasonal = 0.5 + 0.3 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        # Diurnal pattern (higher during day)
        hour = dates.hour + dates.minute / 60
        diurnal = 1 + 0.1 * np.sin(2 * np.pi * (hour - 6) / 24)
        
        # Weekend effect
        is_weekend = (dates.dayofweek >= 5).astype(float)
        weekend = 1 - 0.02 * is_weekend
        
        # Noise
        noise = np.random.lognormal(0, 0.08, n)
        
        # Capacity factor
        cf = seasonal * diurnal * weekend * noise
        cf = np.clip(cf, 0.05, 0.95)
        
        generation = cf * self.capacity_mw
        
        df = pd.DataFrame({"generation_mw": generation}, index=dates)
        df.index.name = "datetime"
        return df

    def fetch_installed_capacity(self, year=None, production_type=None):
        """Return synthetic capacity data."""
        dates = pd.date_range(f"{year or 2024}-01-01", periods=2, freq="YS", tz="UTC")
        df = pd.DataFrame({"capacity_mw": [self.capacity_mw, self.capacity_mw + 50]}, index=dates)
        df.index.name = "datetime"
        return df

    def aggregate_to_hourly(self, quarterly_df, value_col="generation_mw", agg_func="mean"):
        return SmardClient().aggregate_to_hourly(quarterly_df, value_col, agg_func)

    def aggregate_to_weekly(self, hourly_df, value_col="generation_mw", agg_func="mean"):
        return SmardClient().aggregate_to_weekly(hourly_df, value_col, agg_func)

    def get_weekly_capacity_factor(self, start, end, production_type=None, capacity_year=None):
        quarterly = self.fetch_generation(start, end, production_type)
        hourly = self.aggregate_to_hourly(quarterly)
        weekly = self.aggregate_to_weekly(hourly)
        weekly["capacity_factor"] = weekly["generation_mw"] / self.capacity_mw
        weekly["capacity_factor"] = weekly["capacity_factor"].clip(0, 1.2)
        return weekly[["generation_mw", "capacity_factor", "year", "week"]]


def create_smard_client(use_mock: bool = False, region: str = "DE", **kwargs) -> Union[SmardClient, MockSmardClient]:
    """Factory function to create SMARD client."""
    if use_mock:
        return MockSmardClient(region=region, **kwargs)
    return SmardClient(region=region, **kwargs)

# 2025-05-24: Fix SMARD DE hydro data issue: 15-min → weekly CF, module 4007/4008/4009

# 2025-06-19: Add SMARD DE hydro validation: 15-min → weekly CF, module 4007/4008/4009

# 2025-06-20: Bugfix: SMARD API 404 in smard_client - switched to MockSmardClient with DE prox

# 2025-06-21: Bugfix: SMARD API 404 in smard_client - switched to MockSmardClient with DE prox

# 2025-07-24: Add SMARD DE hydro validation: 15-min → weekly CF, module 4007/4008/4009

# 2025-08-09: Fix SMARD API 404: switched to MockSmardClient with DE proxy

# 2025-08-15: Data: 15-min → weekly CF, module 4007/4008/4009 for SMARD DE hydro

# 2025-08-29: Improve SMARD DE hydro ingestion: 15-min → weekly CF, module 4007/4008/4009

# 2025-08-30: Improve SMARD DE hydro ingestion: 15-min → weekly CF, module 4007/4008/4009

# 2025-11-02: Hotfix: SMARD API 404 - switched to MockSmardClient with DE proxy

# 2026-01-04: Fix SMARD API 404: switched to MockSmardClient with DE proxy

# 2026-01-22: Fix SMARD API 404: switched to MockSmardClient with DE proxy

# 2026-05-21: Data: 15-min → weekly CF, module 4007/4008/4009 for SMARD DE hydro

# 2026-06-20: Bugfix: SMARD API 404 in smard_client - switched to MockSmardClient with DE prox

# 2026-06-21: Fix SMARD DE hydro data issue: 15-min → weekly CF, module 4007/4008/4009

# 2026-07-04: Data: 15-min → weekly CF, module 4007/4008/4009 for SMARD DE hydro

# 2026-07-16: Data: 15-min → weekly CF, module 4007/4008/4009 for SMARD DE hydro
