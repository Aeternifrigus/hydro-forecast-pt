"""
Swiss Federal Office of Energy (BFE) / WASTA data client for hydro-forecast-ch.

Fetches:
- Weekly reservoir water levels (total and 4 regions)
- Weekly electricity production by plant type
- WASTA plant registry (all plants >= 300 kW) from admin.ch OGC API

Data published each Wednesday for the previous week.
"""
import io
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import requests

from ..config import get_config
from ..constants import BFE_RESERVOIR_REGIONS

logger = logging.getLogger(__name__)


class BFEClient:
    """
    Client for fetching Swiss hydropower data from BFE (Bundesamt für Energie).

    Data sources:
    - Reservoir levels: Weekly, 4 regions + total, published Wednesdays
    - Production: Weekly electricity production, published Wednesdays
    - WASTA: Plant registry with capacity, location, type from admin.ch OGC API
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize BFE client.

        Args:
            base_url: Base URL for BFE data (default from config)
            cache_dir: Directory for caching downloaded files
        """
        config = get_config()
        self.base_url = base_url or config.bfe_wasta.base_url
        self.reservoir_url = config.bfe_wasta.reservoir_levels_url
        self.production_url = config.bfe_wasta.production_url
        self.wasta_api_url = "https://api.geo.admin.ch/rest/services/ech/MapServer/ch.bfe.statistik-wasserkraftanlagen/FeatureServer/0/query"

        # Cache directory
        self.cache_dir = Path(cache_dir or config.paths.raw_data) / "bfe_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Session for requests
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "hydro-forecast-ch/0.1.0 (research; contact@example.com)"
        })

        logger.info(f"BFEClient initialized, cache={self.cache_dir}")

    # -------------------------------------------------------------------------
    # Reservoir Levels (BFE website - Excel files)
    # -------------------------------------------------------------------------

    def fetch_reservoir_levels(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        regions: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch weekly reservoir water levels.

        BFE publishes weekly reservoir fill levels as percentage of capacity
        for 4 regions and total Switzerland.

        Args:
            start_date: Start date (will find Wednesday of that week)
            end_date: End date
            regions: List of regions (default: all 4 + total)
            use_cache: Use cached files

        Returns:
            DataFrame with columns: date, region, fill_percent, volume_mcm
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        config = get_config()
        target_regions = regions or config.bfe_wasta.reservoir_regions

        cache_key = f"reservoir_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.parquet"
        cache_path = self.cache_dir / cache_key

        if use_cache and cache_path.exists():
            logger.info(f"Loading reservoir levels from cache: {cache_path}")
            return pd.read_parquet(cache_path)

        logger.info(f"Fetching reservoir levels from {start.date()} to {end.date()}")

        all_data = []

        # Generate list of Wednesdays in range
        wednesdays = self._get_wednesdays_in_range(start, end)

        for wed in wednesdays:
            for region in target_regions:
                df = self._fetch_reservoir_week(wed, region)
                if df is not None and not df.empty:
                    all_data.append(df)

        if not all_data:
            logger.warning("No reservoir data fetched")
            return pd.DataFrame(columns=["date", "region", "fill_percent", "volume_mcm"])

        result = pd.concat(all_data, ignore_index=True)
        result = result.sort_values(["date", "region"]).reset_index(drop=True)

        # Save cache
        result.to_parquet(cache_path, index=False)
        logger.info(f"Saved {len(result)} reservoir records to cache")

        return result

    def _get_wednesdays_in_range(self, start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
        """Get all Wednesdays in date range (BFE publishes on Wednesdays)."""
        # Find first Wednesday >= start
        current = start
        while current.weekday() != 2:  # 2 = Wednesday
            current += timedelta(days=1)

        wednesdays = []
        while current <= end:
            wednesdays.append(current)
            current += timedelta(weeks=1)

        return wednesdays

    def _fetch_reservoir_week(self, wednesday: pd.Timestamp, region: str) -> Optional[pd.DataFrame]:
        """
        Fetch reservoir data for a specific week and region.

        Tries multiple URL patterns since BFE doesn't have a documented API.
        """
        year = wednesday.year
        week = wednesday.isocalendar().week

        # Common URL patterns for BFE data
        patterns = [
            f"https://www.bfe.admin.ch/dam/bfe/de/dokumente/statistik/wasserkraft/woechentliche-speicherfuellung/{year}/speicherfuellung-kw{week:02d}-{year}.xlsx",
            f"https://www.bfe.admin.ch/dam/bfe/de/dokumente/statistik/wasserkraft/woechentliche-speicherfuellung/speicherfuellung-kw{week:02d}-{year}.xlsx",
            f"https://www.bfe.admin.ch/bfe/de/home/versorgung/statistik/geodaten/speicherseen/jahr-{year}/kw{week:02d}.xlsx",
        ]

        for url in patterns:
            try:
                resp = self.session.get(url, timeout=30)
                if resp.status_code == 200:
                    return self._parse_reservoir_excel(resp.content, wednesday, region)
            except Exception:
                continue

        logger.debug(f"No reservoir data found for week {week} {year}, region {region}")
        return None

    def _parse_reservoir_excel(
        self, content: bytes, date: pd.Timestamp, region: str
    ) -> Optional[pd.DataFrame]:
        """Parse BFE reservoir Excel file."""
        try:
            # Read all sheets
            xls = pd.read_excel(io.BytesIO(content), sheet_name=None, header=None)

            for sheet_name, df in xls.items():
                # Look for region data
                region_lower = region.lower().replace("_", " ")

                # Try to find the region row
                for idx, row in df.iterrows():
                    row_str = " ".join(str(v).lower() for v in row.values if pd.notna(v))
                    if region_lower in row_str or region in row_str:
                        # Found region row - extract values
                        values = row.values
                        numeric_vals = [v for v in values if isinstance(v, (int, float)) and not pd.isna(v)]

                        if len(numeric_vals) >= 2:
                            fill_pct = numeric_vals[0]
                            volume_mcm = numeric_vals[1]

                            return pd.DataFrame([{
                                "date": date.date(),
                                "region": region,
                                "fill_percent": fill_pct,
                                "volume_mcm": volume_mcm,
                            }])

            return None

        except Exception as e:
            logger.debug(f"Failed to parse reservoir Excel: {e}")
            return None

    # -------------------------------------------------------------------------
    # Weekly Production (BFE website - Excel files)
    # -------------------------------------------------------------------------

    def fetch_weekly_production(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch weekly electricity production by production type.

        BFE publishes weekly production (GWh) for:
        - Hydro run-of-river
        - Hydro reservoir
        - Hydro pumped storage
        - Nuclear
        - Thermal
        - Solar
        - Wind
        - etc.

        Args:
            start_date: Start date
            end_date: End date
            use_cache: Use cached files

        Returns:
            DataFrame with columns: date, production_type, generation_gwh
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        cache_key = f"production_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.parquet"
        cache_path = self.cache_dir / cache_key

        if use_cache and cache_path.exists():
            logger.info(f"Loading production from cache: {cache_path}")
            return pd.read_parquet(cache_path)

        logger.info(f"Fetching weekly production from {start.date()} to {end.date()}")

        all_data = []
        wednesdays = self._get_wednesdays_in_range(start, end)

        for wed in wednesdays:
            df = self._fetch_production_week(wed)
            if df is not None and not df.empty:
                all_data.append(df)

        if not all_data:
            logger.warning("No production data fetched")
            return pd.DataFrame(columns=["date", "production_type", "generation_gwh"])

        result = pd.concat(all_data, ignore_index=True)
        result = result.sort_values(["date", "production_type"]).reset_index(drop=True)

        result.to_parquet(cache_path, index=False)
        logger.info(f"Saved {len(result)} production records to cache")

        return result

    def _fetch_production_week(self, wednesday: pd.Timestamp) -> Optional[pd.DataFrame]:
        """Fetch production data for a specific week."""
        year = wednesday.year
        week = wednesday.isocalendar().week

        patterns = [
            f"https://www.bfe.admin.ch/dam/bfe/de/dokumente/statistik/elektrizitaet/woechentliche-produktion/{year}/produktion-kw{week:02d}-{year}.xlsx",
            f"https://www.bfe.admin.ch/dam/bfe/de/dokumente/statistik/elektrizitaet/woechentliche-produktion/produktion-kw{week:02d}-{year}.xlsx",
        ]

        for url in patterns:
            try:
                resp = self.session.get(url, timeout=30)
                if resp.status_code == 200:
                    return self._parse_production_excel(resp.content, wednesday)
            except Exception:
                continue

        return None

    def _parse_production_excel(self, content: bytes, date: pd.Timestamp) -> Optional[pd.DataFrame]:
        """Parse BFE production Excel file."""
        try:
            xls = pd.read_excel(io.BytesIO(content), sheet_name=None, header=None)

            records = []

            for sheet_name, df in xls.items():
                # Search for production types
                for idx, row in df.iterrows():
                    row_vals = [str(v) for v in row.values if pd.notna(v)]
                    row_str = " ".join(row_vals).lower()

                    # Look for hydro run-of-river
                    if "laufkraftwerk" in row_str or "run-of-river" in row_str or "run of river" in row_str:
                        numeric = [v for v in row.values if isinstance(v, (int, float)) and not pd.isna(v)]
                        if numeric:
                            records.append({
                                "date": date.date(),
                                "production_type": "hydro_run_of_river",
                                "generation_gwh": numeric[0],
                            })

                    # Hydro reservoir
                    if "speicherkraftwerk" in row_str or "reservoir" in row_str:
                        numeric = [v for v in row.values if isinstance(v, (int, float)) and not pd.isna(v)]
                        if numeric:
                            records.append({
                                "date": date.date(),
                                "production_type": "hydro_reservoir",
                                "generation_gwh": numeric[0],
                            })

                    # Pumped storage
                    if "pumpspeicher" in row_str or "pumped" in row_str:
                        numeric = [v for v in row.values if isinstance(v, (int, float)) and not pd.isna(v)]
                        if numeric:
                            records.append({
                                "date": date.date(),
                                "production_type": "hydro_pumped_storage",
                                "generation_gwh": numeric[0],
                            })

            if records:
                return pd.DataFrame(records)
            return None

        except Exception as e:
            logger.debug(f"Failed to parse production Excel: {e}")
            return None

    # -------------------------------------------------------------------------
    # WASTA Plant Registry (admin.ch OGC API)
    # -------------------------------------------------------------------------

    def fetch_wasta_plants(
        self,
        year: Optional[int] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch WASTA plant registry (Wasserkraftanlagen-Statistik) from admin.ch OGC API.

        Contains all hydropower plants >= 300 kW with:
        - Plant name, location, canton
        - Installed capacity (MW)
        - Type (run-of-river, reservoir, pumped storage)
        - Coordinates
        - Owner/operator

        Uses admin.ch API endpoint:
        https://api.geo.admin.ch/rest/services/ech/MapServer/ch.bfe.statistik-wasserkraftanlagen/FeatureServer/0/query

        Args:
            year: Year of registry (not used for API, kept for compatibility)
            use_cache: Use cached file

        Returns:
            DataFrame with plant information
        """
        year = year or datetime.now().year

        cache_key = f"wasta_plants_{year}.parquet"
        cache_path = self.cache_dir / cache_key

        if use_cache and cache_path.exists():
            logger.info(f"Loading WASTA plants from cache: {cache_path}")
            return pd.read_parquet(cache_path)

        logger.info(f"Fetching WASTA plant registry from admin.ch API")

        try:
            df = self._fetch_wasta_from_admin_ch()
            if df is not None and not df.empty:
                df.to_parquet(cache_path, index=False)
                logger.info(f"Saved {len(df)} WASTA plants to cache")
                return df
        except Exception as e:
            logger.error(f"Failed to fetch WASTA from admin.ch: {e}")

        # Fallback to BFE website Excel
        logger.info("Falling back to BFE website for WASTA data")
        return self._fetch_wasta_from_bfe_website(year, use_cache, cache_path)

    def _fetch_wasta_from_admin_ch(self) -> Optional[pd.DataFrame]:
        """
        Fetch WASTA plant data from admin.ch OGC Feature Service API.

        Query Parameters:
        - layer=ch.bfe.statistik-wasserkraftanlagen
        - searchText=*
        - searchField=name
        - returnGeometry=true
        - f=json (or geojson)
        """
        # OGC API Feature Service query
        params = {
            "where": "1=1",  # All features
            "outFields": "*",
            "returnGeometry": "true",
            "f": "geojson",
            "resultRecordCount": 10000,  # Max records
        }

        try:
            resp = self.session.get(self.wasta_api_url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            return self._parse_wasta_geojson(data)

        except Exception as e:
            logger.error(f"Admin.ch API request failed: {e}")
            return None

    def _parse_wasta_geojson(self, geojson: dict) -> pd.DataFrame:
        """Parse WASTA GeoJSON response from admin.ch API."""
        features = geojson.get("features", [])
        if not features:
            logger.warning("No features in WASTA GeoJSON response")
            return pd.DataFrame()

        records = []
        for feat in features:
            props = feat.get("properties", {})
            geom = feat.get("geometry", {})

            # Extract coordinates
            coords = geom.get("coordinates", [None, None])
            lon, lat = coords[0], coords[1] if len(coords) >= 2 else (None, None)

            # Map admin.ch field names to standard names
            # Based on ch.bfe.statistik-wasserkraftanlagen schema
            record = {
                "name": props.get("name") or props.get("anlage_name") or props.get("kraftwerk"),
                "canton": props.get("kanton") or props.get("canton") or props.get("kt"),
                "capacity_mw": props.get("leistung_mw") or props.get("capacity_mw") or props.get("installierte_leistung"),
                "type": props.get("typ") or props.get("type") or props.get("anlage_typ"),
                "latitude": lat,
                "longitude": lon,
                "municipality": props.get("gemeinde") or props.get("municipality") or props.get("ort"),
                "river": props.get("gewasser") or props.get("river") or props.get("fluss"),
                "owner": props.get("betreiber") or props.get("owner") or props.get("eigentuemer"),
                "elevation_m": props.get("hoehe_m") or props.get("elevation_m"),
                "commissioning_year": props.get("inbetriebnahme") or props.get("baujahr"),
                "object_id": props.get("objectid") or props.get("id"),
            }
            records.append(record)

        df = pd.DataFrame(records)

        # Normalize type field
        if "type" in df.columns:
            df["type"] = df["type"].apply(self._normalize_plant_type)

        # Ensure numeric capacity
        if "capacity_mw" in df.columns:
            df["capacity_mw"] = pd.to_numeric(df["capacity_mw"], errors="coerce")

        logger.info(f"Parsed {len(df)} plants from admin.ch WASTA API")
        return df

    def _normalize_plant_type(self, type_val: str) -> str:
        """Normalize plant type to standard categories."""
        if pd.isna(type_val):
            return "unknown"
        type_lower = str(type_val).lower()
        if "lauf" in type_lower or "run.of.river" in type_lower:
            return "run_of_river"
        elif "speicher" in type_lower or "reservoir" in type_lower:
            return "reservoir"
        elif "pump" in type_lower:
            return "pumped_storage"
        elif "kraftwerk" in type_lower:
            return "unknown"
        return "unknown"

    def _fetch_wasta_from_bfe_website(
        self,
        year: int,
        use_cache: bool,
        cache_path: Path,
    ) -> pd.DataFrame:
        """Fallback: Fetch WASTA from BFE website Excel."""
        patterns = [
            f"https://www.bfe.admin.ch/dam/bfe/de/dokumente/statistik/wasserkraft/wasta/wasta-{year}.xlsx",
            f"https://www.bfe.admin.ch/bfe/de/home/versorgung/statistik/wasta/jahr-{year}.xlsx",
        ]

        for url in patterns:
            try:
                resp = self.session.get(url, timeout=60)
                if resp.status_code == 200:
                    df = self._parse_wasta_excel(resp.content, year)
                    if df is not None and not df.empty:
                        if use_cache:
                            df.to_parquet(cache_path, index=False)
                        return df
            except Exception:
                continue

        logger.warning("Could not fetch WASTA data from any source, returning empty DataFrame")
        return pd.DataFrame()

    def _parse_wasta_excel(self, content: bytes, year: int) -> Optional[pd.DataFrame]:
        """Parse WASTA plant registry Excel."""
        try:
            xls = pd.read_excel(io.BytesIO(content), sheet_name=None)

            # Usually first sheet has the data
            for sheet_name, df in xls.items():
                if len(df) > 10:  # Likely the data sheet
                    # Standardize column names
                    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

                    # Map common column names
                    col_map = {
                        "name": ["name", "anlage", "plant_name", "kraftwerk"],
                        "canton": ["kanton", "canton", "kt"],
                        "capacity_mw": ["leistung_mw", "capacity_mw", "installierte_leistung_mw", "mw"],
                        "type": ["typ", "type", "art", "anlage_typ"],
                        "latitude": ["lat", "latitude", "y_coord", "nord"],
                        "longitude": ["lon", "lng", "longitude", "x_coord", "ost"],
                        "municipality": ["gemeinde", "municipality", "ort"],
                        "river": ["gewasser", "river", "fluss"],
                        "owner": ["betreiber", "owner", "eigentuemer"],
                    }

                    # Find matching columns
                    mapped = {}
                    for std_col, candidates in col_map.items():
                        for cand in candidates:
                            if cand in df.columns:
                                mapped[std_col] = cand
                                break

                    if "name" in mapped and "capacity_mw" in mapped:
                        result = df[list(mapped.values())].copy()
                        result.columns = list(mapped.keys())
                        result["year"] = year
                        result["type"] = result["type"].apply(self._normalize_plant_type)
                        return result

            return None

        except Exception as e:
            logger.error(f"Failed to parse WASTA Excel: {e}")
            return None

    # -------------------------------------------------------------------------
    # Convenience Methods
    # -------------------------------------------------------------------------

    def get_run_of_river_capacity(self, wasta_df: Optional[pd.DataFrame] = None) -> float:
        """Get total installed run-of-river capacity in MW."""
        if wasta_df is None:
            wasta_df = self.fetch_wasta_plants()

        if wasta_df.empty:
            return 0.0

        run_of_river = wasta_df[wasta_df["type"] == "run_of_river"]
        return run_of_river["capacity_mw"].sum()

    def get_reservoir_capacity(self, wasta_df: Optional[pd.DataFrame] = None) -> float:
        """Get total installed reservoir capacity in MW."""
        if wasta_df is None:
            wasta_df = self.fetch_wasta_plants()

        if wasta_df.empty:
            return 0.0

        reservoir = wasta_df[wasta_df["type"] == "reservoir"]
        return reservoir["capacity_mw"].sum()

    def get_basin_plants(self, basin: str, wasta_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Filter plants by basin using basin outlet proximity."""
        from ..constants import BASIN_OUTLETS, BASIN_BOUNDS

        if wasta_df is None:
            wasta_df = self.fetch_wasta_plants()

        if wasta_df.empty or basin not in BASIN_BOUNDS:
            return pd.DataFrame()

        bounds = BASIN_BOUNDS[basin]
        min_lon, min_lat, max_lon, max_lat = bounds

        return wasta_df[
            (wasta_df["longitude"] >= min_lon) & (wasta_df["longitude"] <= max_lon) &
            (wasta_df["latitude"] >= min_lat) & (wasta_df["latitude"] <= max_lat)
        ].copy()


class MockBFEClient:
    """Mock BFE client for testing without network access."""

    def __init__(self, **kwargs):
        self.cache_dir = Path(kwargs.get("cache_dir", "/tmp/bfe_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Using MockBFEClient for testing")

    def fetch_reservoir_levels(self, start_date, end_date, regions=None, use_cache=True):
        """Generate synthetic reservoir data."""
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        target_regions = regions or ["all", "region_1", "region_2", "region_3", "region_4"]

        wednesdays = []
        current = start
        while current.weekday() != 2:
            current += timedelta(days=1)
        while current <= end:
            wednesdays.append(current)
            current += timedelta(weeks=1)

        import numpy as np
        np.random.seed(42)

        records = []
        for wed in wednesdays:
            day_of_year = wed.dayofyear
            seasonal = 0.6 + 0.3 * np.sin(2 * np.pi * (day_of_year - 60) / 365)

            for region in target_regions:
                noise = np.random.normal(1, 0.05)
                fill_pct = np.clip(seasonal * noise * 100, 20, 95)
                volume = fill_pct * 10

                week_end = (wed + timedelta(days=4)).date()  # Sunday after Wednesday
                records.append({
                    "date": week_end,
                    "week_end": week_end,
                    "region": region,
                    "fill_percent": fill_pct,
                    "volume_mcm": volume,
                    "year": pd.Timestamp(week_end).year,
                    "week": pd.Timestamp(week_end).isocalendar().week,
                })

        return pd.DataFrame(records)

    def fetch_weekly_production(self, start_date, end_date, use_cache=True):
        """Generate synthetic production data."""
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        wednesdays = []
        current = start
        while current.weekday() != 2:
            current += timedelta(days=1)
        while current <= end:
            wednesdays.append(current)
            current += timedelta(weeks=1)

        import numpy as np
        np.random.seed(123)

        records = []
        for wed in wednesdays:
            day_of_year = wed.dayofyear
            seasonal = 0.5 + 0.3 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

            week_end = (wed + timedelta(days=4)).date()  # Sunday after Wednesday
            
            ror = seasonal * np.random.lognormal(0, 0.1) * 500
            records.append({
                "date": week_end,
                "week_end": week_end,
                "production_type": "hydro_run_of_river",
                "generation_gwh": ror * 1000,
                "year": pd.Timestamp(week_end).year,
                "week": pd.Timestamp(week_end).isocalendar().week,
            })

            res = (1 - seasonal * 0.5) * np.random.lognormal(0, 0.15) * 800
            records.append({
                "date": week_end,
                "week_end": week_end,
                "production_type": "hydro_reservoir",
                "generation_gwh": res * 1000,
                "year": pd.Timestamp(week_end).year,
                "week": pd.Timestamp(week_end).isocalendar().week,
            })

            pumped = 0.1 * seasonal * np.random.lognormal(0, 0.1) * 200
            records.append({
                "date": week_end,
                "week_end": week_end,
                "production_type": "hydro_pumped_storage",
                "generation_gwh": pumped * 1000,
                "year": pd.Timestamp(week_end).year,
                "week": pd.Timestamp(week_end).isocalendar().week,
            })

        return pd.DataFrame(records)

    def fetch_wasta_plants(self, year=None, use_cache=True):
        """Return synthetic WASTA plant data."""
        import numpy as np
        np.random.seed(42)

        plants = [
            {"name": "Grande Dixence", "canton": "VS", "capacity_mw": 1800, "type": "reservoir",
             "latitude": 46.08, "longitude": 7.41, "municipality": "Hérémence", "river": "Dixence", "basin_id": "rhone"},
            {"name": "Mauvoisin", "canton": "VS", "capacity_mw": 280, "type": "reservoir",
             "latitude": 46.02, "longitude": 7.35, "municipality": "Bagnes", "river": "Dranse de Bagnes", "basin_id": "rhone"},
            {"name": "Laufenburg", "canton": "AG", "capacity_mw": 30, "type": "run_of_river",
             "latitude": 47.56, "longitude": 8.06, "municipality": "Laufenburg", "river": "Rhine", "basin_id": "rhine"},
            {"name": "Rheinau", "canton": "ZH", "capacity_mw": 60, "type": "run_of_river",
             "latitude": 47.56, "longitude": 8.61, "municipality": "Rheinau", "river": "Rhine", "basin_id": "rhine"},
            {"name": "Verbois", "canton": "GE", "capacity_mw": 120, "type": "run_of_river",
             "latitude": 46.19, "longitude": 6.00, "municipality": "Versoix", "river": "Rhône", "basin_id": "rhone"},
        ]

        df = pd.DataFrame(plants)
        df["year"] = year or 2024
        return df

    def get_run_of_river_capacity(self, wasta_df=None):
        if wasta_df is None:
            wasta_df = self.fetch_wasta_plants()
        return wasta_df[wasta_df["type"] == "run_of_river"]["capacity_mw"].sum()

    def get_reservoir_capacity(self, wasta_df=None):
        if wasta_df is None:
            wasta_df = self.fetch_wasta_plants()
        return wasta_df[wasta_df["type"] == "reservoir"]["capacity_mw"].sum()

    def get_basin_plants(self, basin: str, wasta_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        from ..constants import BASIN_BOUNDS
        if wasta_df is None:
            wasta_df = self.fetch_wasta_plants()
        if wasta_df.empty or basin not in BASIN_BOUNDS:
            return pd.DataFrame()
        bounds = BASIN_BOUNDS[basin]
        min_lon, min_lat, max_lon, max_lat = bounds
        return wasta_df[
            (wasta_df["longitude"] >= min_lon) & (wasta_df["longitude"] <= max_lon) &
            (wasta_df["latitude"] >= min_lat) & (wasta_df["latitude"] <= max_lat)
        ].copy()


def create_bfe_client(use_mock: bool = False, **kwargs) -> Union[BFEClient, MockBFEClient]:
    """Factory function to create BFE client."""
    if use_mock:
        return MockBFEClient(**kwargs)
    return BFEClient(**kwargs)
# 2025-04-03: Fix BFE production data issue: weekly by type: RoR/reservoir/pumped

# 2025-04-06: Add BFE reservoir validation: weekly Excel, multi-sheet, fill levels

# 2025-04-18: Improve BFE reservoir ingestion: weekly Excel, multi-sheet, fill levels

# 2025-06-12: Data: weekly Excel, multi-sheet, fill levels for BFE reservoir

# 2025-06-15: Add BFE production validation: weekly by type: RoR/reservoir/pumped

# 2025-06-21: Fix BFE production data issue: weekly by type: RoR/reservoir/pumped

# 2025-07-06: Data: weekly by type: RoR/reservoir/pumped for BFE production

# 2025-07-12: Data: weekly by type: RoR/reservoir/pumped for BFE production

# 2025-07-12: Fix BFE production data issue: weekly by type: RoR/reservoir/pumped

# 2025-07-17: Fix BFE Excel 404: MockBFEClient with seasonal cycles

# 2025-07-19: Hotfix: BFE Excel 404 - MockBFEClient with seasonal cycles

# 2025-08-09: Fix BFE production data issue: weekly by type: RoR/reservoir/pumped

# 2025-08-10: Add BFE reservoir validation: weekly Excel, multi-sheet, fill levels

# 2025-08-15: Improve BFE production ingestion: weekly by type: RoR/reservoir/pumped

# 2025-08-31: Improve BFE production ingestion: weekly by type: RoR/reservoir/pumped

# 2025-12-28: Hotfix: BFE Excel 404 - MockBFEClient with seasonal cycles

# 2026-05-08: Data: weekly by type: RoR/reservoir/pumped for BFE production

# 2026-05-31: Data: weekly Excel, multi-sheet, fill levels for BFE reservoir

# 2026-06-07: Data: weekly Excel, multi-sheet, fill levels for BFE reservoir

# 2026-06-14: Data: weekly Excel, multi-sheet, fill levels for BFE reservoir

# 2026-06-18: Fix BFE reservoir data issue: weekly Excel, multi-sheet, fill levels

# 2026-06-25: Hotfix: BFE Excel 404 - MockBFEClient with seasonal cycles

# 2026-06-27: Add BFE production validation: weekly by type: RoR/reservoir/pumped

# 2026-07-04: Improve BFE reservoir ingestion: weekly Excel, multi-sheet, fill levels

# 2026-07-05: Improve BFE production ingestion: weekly by type: RoR/reservoir/pumped

# 2026-07-09: Fix BFE production data issue: weekly by type: RoR/reservoir/pumped

# 2026-07-18: Data: weekly by type: RoR/reservoir/pumped for BFE production

# 2026-07-19: Fix: BFE Excel 404 - MockBFEClient with seasonal cycles
