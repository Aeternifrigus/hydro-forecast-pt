"""
Hydrological feature engineering for hydro-forecast-ch.

Computes:
- Rolling precipitation sums (1, 2, 3, 4, 8, 12, 26 weeks)
- Antecedent Wetness Index (AWI) with exponential decay
- Temperature degree days (heating, cooling, snowmelt)
- Snow accumulation and melt features
- Soil moisture proxy features
"""
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..config import get_config

logger = logging.getLogger(__name__)


def compute_rolling_precip(
    df: pd.DataFrame,
    precip_col: str = "precipitation_mm",
    date_col: str = "week_end",
    windows_weeks: Optional[List[int]] = None,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute rolling precipitation sums over multiple windows.

    Args:
        df: DataFrame with precipitation data
        precip_col: Precipitation column name
        date_col: Date column for sorting
        windows_weeks: List of window sizes in weeks
        group_col: Column to group by (e.g., basin_id)

    Returns:
        DataFrame with added rolling precipitation columns
    """
    config = get_config()
    windows = windows_weeks or config.features.precipitation.rolling_windows_weeks

    result = df.copy()
    result = result.sort_values([group_col, date_col] if group_col else [date_col])

    if group_col:
        grouped = result.groupby(group_col)
    else:
        grouped = [("", result)]

    for window in windows:
        col_name = f"precip_roll_{window}w"

        if group_col:
            result[col_name] = grouped[precip_col].transform(
                lambda x: x.rolling(window, min_periods=1).sum()
            )
        else:
            result[col_name] = result[precip_col].rolling(window, min_periods=1).sum()

        logger.debug(f"Computed {col_name}: mean={result[col_name].mean():.1f}")

    return result


def compute_awi(
    df: pd.DataFrame,
    precip_col: str = "precipitation_mm",
    date_col: str = "week_end",
    decay_weeks: Optional[int] = None,
    group_col: Optional[str] = None,
    awi_col: str = "awi",
) -> pd.DataFrame:
    """
    Compute Antecedent Wetness Index (AWI) with exponential decay.

    AWI_t = Σ_{i=0}^{t} P_i × exp(-(t-i)/τ)

    Where τ is the decay time constant in weeks.

    Args:
        df: DataFrame with precipitation data
        precip_col: Precipitation column name
        date_col: Date column for sorting
        decay_weeks: Decay time constant (weeks)
        group_col: Column to group by
        awi_col: Output column name

    Returns:
        DataFrame with added AWI column
    """
    config = get_config()
    tau = decay_weeks or config.features.precipitation.awi_decay_weeks

    result = df.copy()
    result = result.sort_values([group_col, date_col] if group_col else [date_col])

    decay_factor = np.exp(-1.0 / tau)

    if group_col:
        grouped = result.groupby(group_col)
    else:
        grouped = [("", result)]

    awi_values = []

    for _, group in grouped:
        precip = group[precip_col].values
        awi = np.zeros_like(precip, dtype=float)

        # Recursive computation: AWI_t = P_t + decay_factor * AWI_{t-1}
        for i in range(len(precip)):
            if i == 0:
                awi[i] = precip[i]
            else:
                awi[i] = precip[i] + decay_factor * awi[i-1]

        awi_values.extend(awi)

    result[awi_col] = awi_values
    logger.debug(f"Computed AWI (τ={tau}w): mean={result[awi_col].mean():.1f}")

    return result


def compute_degree_days(
    df: pd.DataFrame,
    temp_col: str = "temperature_c",
    date_col: str = "week_end",
    base_heating: float = 15.0,
    base_cooling: float = 20.0,
    base_snowmelt: float = 0.0,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute degree days for heating, cooling, and snowmelt.

    Weekly degree days = Σ daily degree days within the week.
    Since we have weekly data, we approximate:
    - HDD = max(0, base_heating - temp_mean) * 7
    - CDD = max(0, temp_mean - base_cooling) * 7
    - SDD (snowmelt) = max(0, temp_mean - base_snowmelt) * 7

    Args:
        df: DataFrame with temperature data
        temp_col: Temperature column name
        date_col: Date column
        base_heating: Heating degree day base (°C)
        base_cooling: Cooling degree day base (°C)
        base_snowmelt: Snowmelt degree day base (°C)
        group_col: Column to group by

    Returns:
        DataFrame with added degree day columns
    """
    result = df.copy()

    # Weekly HDD/CDD/SDD (approximate from weekly mean temp)
    result["hdd_week"] = np.maximum(0, base_heating - result[temp_col]) * 7
    result["cdd_week"] = np.maximum(0, result[temp_col] - base_cooling) * 7
    result["sdd_week"] = np.maximum(0, result[temp_col] - base_snowmelt) * 7

    # Also compute rolling degree days
    if group_col:
        for window in [2, 4, 8, 12]:
            result[f"hdd_roll_{window}w"] = result.groupby(group_col)["hdd_week"].transform(
                lambda x: x.rolling(window, min_periods=1).sum()
            )
            result[f"sdd_roll_{window}w"] = result.groupby(group_col)["sdd_week"].transform(
                lambda x: x.rolling(window, min_periods=1).sum()
            )
    else:
        for window in [2, 4, 8, 12]:
            result[f"hdd_roll_{window}w"] = result["hdd_week"].rolling(window, min_periods=1).sum()
            result[f"sdd_roll_{window}w"] = result["sdd_week"].rolling(window, min_periods=1).sum()

    logger.debug("Computed degree days (HDD, CDD, SDD)")
    return result


def compute_snow_features(
    df: pd.DataFrame,
    snowfall_col: str = "snowfall_mm",
    snow_depth_col: str = "snow_depth_cm",
    temp_col: str = "temperature_c",
    date_col: str = "week_end",
    group_col: Optional[str] = None,
    melt_factor: float = 3.5,  # mm/°C/day
) -> pd.DataFrame:
    """
    Compute snow-related features.

    Features:
    - Snow water equivalent (SWE) from snowfall accumulation
    - Snowmelt estimate from degree-day method
    - Snow cover fraction proxy
    - Snow accumulation/melt rates

    Args:
        df: DataFrame with snow and temperature data
        snowfall_col: Snowfall column (mm water equivalent)
        snow_depth_col: Snow depth column (cm)
        temp_col: Temperature column (°C)
        date_col: Date column
        group_col: Grouping column
        melt_factor: Degree-day melt factor (mm/°C/day)

    Returns:
        DataFrame with added snow features
    """
    result = df.copy()
    result = result.sort_values([group_col, date_col] if group_col else [date_col])

    # Snow accumulation (cumulative snowfall since last melt)
    if group_col:
        result["swe_cumulative"] = result.groupby(group_col)[snowfall_col].transform(
            lambda x: x.cumsum()
        )
    else:
        result["swe_cumulative"] = result[snowfall_col].cumsum()

    # Snowmelt estimate (degree-day method)
    # Melt = melt_factor * max(0, temp - 0°C) * 7 days
    melt_per_week = melt_factor * np.maximum(0, result[temp_col]) * 7
    result["snowmelt_mm_week"] = melt_per_week

    # Net SWE change
    result["swe_net_change"] = result[snowfall_col] - result["snowmelt_mm_week"]

    # Snow cover proxy (binary: snow depth > 0)
    result["snow_cover"] = (result[snow_depth_col] > 0).astype(int)

    # Snow depth change
    if group_col:
        result["snow_depth_change"] = result.groupby(group_col)[snow_depth_col].transform(
            lambda x: x.diff()
        )
    else:
        result["snow_depth_change"] = result[snow_depth_col].diff()

    # Rolling snowmelt (important for spring flood forecasting)
    for window in [2, 4, 8]:
        col_name = f"snowmelt_roll_{window}w"
        if group_col:
            result[col_name] = result.groupby(group_col)["snowmelt_mm_week"].transform(
                lambda x: x.rolling(window, min_periods=1).sum()
            )
        else:
            result[col_name] = result["snowmelt_mm_week"].rolling(window, min_periods=1).sum()

    # Snow depletion indicator (when melt > snowfall)
    result["snow_depleting"] = (result["snowmelt_mm_week"] > result[snowfall_col]).astype(int)

    logger.debug("Computed snow features (SWE, melt, cover, depletion)")
    return result


def compute_soil_moisture_proxy(
    df: pd.DataFrame,
    precip_col: str = "precipitation_mm",
    temp_col: str = "temperature_c",
    date_col: str = "week_end",
    group_col: Optional[str] = None,
    field_capacity: float = 200.0,  # mm
    wilting_point: float = 50.0,   # mm
) -> pd.DataFrame:
    """
    Compute soil moisture proxy using a simple bucket model.

    Two-layer bucket:
    - Fast layer (surface): quick response, low capacity
    - Slow layer (root zone): slow response, high capacity

    This is a "physics-lite" feature for ML models.

    Args:
        df: DataFrame with precip and temp
        precip_col: Precipitation column
        temp_col: Temperature column (for ET estimation)
        date_col: Date column
        group_col: Grouping column
        field_capacity: Maximum soil moisture (mm)
        wilting_point: Wilting point (mm)

    Returns:
        DataFrame with soil moisture features
    """
    result = df.copy()
    result = result.sort_values([group_col, date_col] if group_col else [date_col])

    # Simple potential ET estimate (mm/week)
    # Using temperature-based approximation
    pet = np.maximum(0, (result[temp_col] - 5) * 0.5 * 7)  # Very rough
    result["pet_week"] = pet

    # Two-layer bucket state
    fast_capacity = 50.0  # mm
    slow_capacity = field_capacity - fast_capacity

    def bucket_step(group):
        fast_store = fast_capacity * 0.5  # Start at 50%
        slow_store = slow_capacity * 0.5

        fast_vals = []
        slow_vals = []
        percolation_vals = []
        total_vals = []

        for _, row in group.iterrows():
            p = row[precip_col]
            et = row["pet_week"]

            # Fast layer receives all precipitation
            fast_store = min(fast_store + p, fast_capacity)

            # ET from fast layer first
            fast_et = min(et, fast_store)
            fast_store -= fast_et
            et_remaining = et - fast_et

            # Remaining ET from slow layer
            slow_et = min(et_remaining, slow_store)
            slow_store -= slow_et

            # Percolation from fast to slow
            perc = max(0, fast_store - fast_capacity * 0.3) * 0.2
            fast_store -= perc
            slow_store = min(slow_store + perc, slow_capacity)

            fast_vals.append(fast_store)
            slow_vals.append(slow_store)
            percolation_vals.append(perc)
            total_vals.append(fast_store + slow_store)

        # Return a DataFrame with the same index as the input group
        return pd.DataFrame({
            "fast_soil_moisture": fast_vals,
            "slow_soil_moisture": slow_vals,
            "percolation": percolation_vals,
            "total_soil_moisture": total_vals,
        }, index=group.index)

    if group_col:
        bucket_results = result.groupby(group_col).apply(bucket_step, include_groups=False)
        for col in bucket_results.columns:
            result[col] = bucket_results[col].values
    else:
        bucket_results = bucket_step(result)
        for col in bucket_results.columns:
            result[col] = bucket_results[col].values

    # Normalized soil moisture (0-1)
    result["soil_moisture_norm"] = (
        result["total_soil_moisture"] / field_capacity
    ).clip(0, 1)

    # Saturation excess indicator
    result["saturation_excess"] = (result["total_soil_moisture"] > field_capacity * 0.9).astype(int)

    logger.debug("Computed soil moisture proxy (fast/slow bucket)")
    return result


def compute_hydro_features(
    df: pd.DataFrame,
    precip_col: str = "precipitation_mm",
    temp_col: str = "temperature_c",
    snowfall_col: str = "snowfall_mm",
    snow_depth_col: str = "snow_depth_cm",
    date_col: str = "week_end",
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute all hydrological features in one call.

    This is the main feature engineering function.

    Args:
        df: Input DataFrame with weather data
        precip_col: Precipitation column
        temp_col: Temperature column
        snowfall_col: Snowfall column
        snow_depth_col: Snow depth column
        date_col: Date column
        group_col: Grouping column (basin_id)

    Returns:
        DataFrame with all hydro features added
    """
    logger.info(f"Computing hydro features for {len(df)} records")

    result = df.copy()

    # 1. Rolling precipitation
    result = compute_rolling_precip(
        result, precip_col, date_col, group_col=group_col
    )

    # 2. AWI
    result = compute_awi(
        result, precip_col, date_col, group_col=group_col
    )

    # 3. Degree days
    result = compute_degree_days(
        result, temp_col, date_col, group_col=group_col
    )

    # 4. Snow features
    if snowfall_col in result.columns and snow_depth_col in result.columns:
        result = compute_snow_features(
            result, snowfall_col, snow_depth_col, temp_col, date_col, group_col
        )

    # 5. Soil moisture proxy
    result = compute_soil_moisture_proxy(
        result, precip_col, temp_col, date_col, group_col
    )

    logger.info(f"Hydro features computed: {len(result.columns)} columns")
    return result


# Convenience functions for specific features
def get_precipitation_features(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Compute only precipitation features."""
    return compute_rolling_precip(df, **kwargs)


def get_awi_feature(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Compute only AWI feature."""
    return compute_awi(df, **kwargs)


def get_degree_day_features(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Compute only degree day features."""
    return compute_degree_days(df, **kwargs)


def get_snow_features(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Compute only snow features."""
    return compute_snow_features(df, **kwargs)


def get_soil_moisture_features(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Compute only soil moisture features."""
    return compute_soil_moisture_proxy(df, **kwargs)