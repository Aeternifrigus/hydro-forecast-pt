"""
Temporal feature engineering for hydro-forecast-ch.

Computes:
- Fourier terms for weekly seasonality
- Lag features (autoregressive)
- Holiday/calendar features
- Week-of-year, month, quarter indicators
"""
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..config import get_config

logger = logging.getLogger(__name__)


def add_seasonal_fourier(
    df: pd.DataFrame,
    date_col: str = "week_end",
    period_weeks: float = 52.1429,  # Weeks per year
    n_terms: int = 4,
    prefix: str = "fourier",
) -> pd.DataFrame:
    """
    Add Fourier terms for seasonal patterns.

    Fourier series: sin(2πkt/P), cos(2πkt/P) for k=1..n_terms

    Args:
        df: DataFrame with date column
        date_col: Date column name
        period_weeks: Period in weeks (52.1429 for annual)
        n_terms: Number of Fourier term pairs
        prefix: Column prefix

    Returns:
        DataFrame with Fourier term columns
    """
    result = df.copy()

    # Convert date to continuous week number
    dates = pd.to_datetime(result[date_col])
    # Week of year (1-52/53)
    week_of_year = dates.dt.isocalendar().week
    year = dates.dt.year

    # Continuous time in weeks since epoch
    # Use year + week/52 for smooth transitions
    t = year + (week_of_year - 1) / period_weeks

    for k in range(1, n_terms + 1):
        angle = 2 * np.pi * k * t / period_weeks
        result[f"{prefix}_sin_{k}"] = np.sin(angle)
        result[f"{prefix}_cos_{k}"] = np.cos(angle)

    logger.debug(f"Added {n_terms} Fourier term pairs for period {period_weeks} weeks")
    return result


def add_multiple_seasonalities(
    df: pd.DataFrame,
    date_col: str = "week_end",
    seasonalities: Optional[Dict[str, Tuple[float, int]]] = None,
) -> pd.DataFrame:
    """
    Add Fourier terms for multiple seasonalities.

    Args:
        df: DataFrame with date column
        date_col: Date column name
        seasonalities: Dict of {name: (period_weeks, n_terms)}

    Returns:
        DataFrame with Fourier terms for each seasonality
    """
    default_seasonalities = {
        "annual": (52.1429, 4),
        "semi_annual": (26.07, 2),
        "quarterly": (13.04, 2),
    }

    seasonalities = seasonalities or default_seasonalities

    result = df.copy()
    for name, (period, n_terms) in seasonalities.items():
        result = add_seasonal_fourier(
            result, date_col, period, n_terms, prefix=f"fourier_{name}"
        )

    return result


def add_week_indicators(
    df: pd.DataFrame,
    date_col: str = "week_end",
    prefix: str = "week",
) -> pd.DataFrame:
    """
    Add week-of-year, month, quarter indicators.

    Args:
        df: DataFrame with date column
        date_col: Date column name
        prefix: Column prefix

    Returns:
        DataFrame with calendar indicators
    """
    result = df.copy()
    dates = pd.to_datetime(result[date_col])

    result[f"{prefix}_of_year"] = dates.dt.isocalendar().week
    result[f"{prefix}_month"] = dates.dt.month
    result[f"{prefix}_quarter"] = dates.dt.quarter
    result[f"{prefix}_year"] = dates.dt.year

    # Cyclical encoding for week of year
    result[f"{prefix}_sin"] = np.sin(2 * np.pi * result[f"{prefix}_of_year"] / 52)
    result[f"{prefix}_cos"] = np.cos(2 * np.pi * result[f"{prefix}_of_year"] / 52)

    # Month cyclical
    result[f"{prefix}_month_sin"] = np.sin(2 * np.pi * result[f"{prefix}_month"] / 12)
    result[f"{prefix}_month_cos"] = np.cos(2 * np.pi * result[f"{prefix}_month"] / 12)

    return result


def add_lag_features(
    df: pd.DataFrame,
    target_cols: List[str],
    lags: List[int] = [1, 2, 3, 4, 8, 12, 26, 52],
    group_col: Optional[str] = None,
    prefix: str = "lag",
) -> pd.DataFrame:
    """
    Add lag features for autoregressive modeling.

    Args:
        df: DataFrame
        target_cols: Columns to create lags for
        lags: List of lag periods (in weeks)
        group_col: Column to group by (e.g., basin_id)
        prefix: Column prefix

    Returns:
        DataFrame with lag columns
    """
    result = df.copy()
    result = result.sort_values([group_col, "week_end"] if group_col else ["week_end"])

    for col in target_cols:
        if col not in result.columns:
            logger.warning(f"Column {col} not found, skipping lag features")
            continue

        for lag in lags:
            lag_col = f"{prefix}_{col}_{lag}w"

            if group_col:
                result[lag_col] = result.groupby(group_col)[col].shift(lag)
            else:
                result[lag_col] = result[col].shift(lag)

    logger.debug(f"Added lag features for {len(target_cols)} columns, lags={lags}")
    return result


def add_rolling_stats(
    df: pd.DataFrame,
    target_cols: List[str],
    windows: List[int] = [2, 4, 8, 12, 26],
    stats: List[str] = ["mean", "std", "min", "max"],
    group_col: Optional[str] = None,
    prefix: str = "roll",
) -> pd.DataFrame:
    """
    Add rolling window statistics.

    Args:
        df: DataFrame
        target_cols: Columns to compute rolling stats for
        windows: Window sizes (weeks)
        stats: Statistics to compute
        group_col: Grouping column
        prefix: Column prefix

    Returns:
        DataFrame with rolling statistics
    """
    result = df.copy()
    result = result.sort_values([group_col, "week_end"] if group_col else ["week_end"])

    for col in target_cols:
        if col not in result.columns:
            continue

        for window in windows:
            for stat in stats:
                col_name = f"{prefix}_{stat}_{col}_{window}w"

                if group_col:
                    result[col_name] = result.groupby(group_col)[col].transform(
                        lambda x: getattr(x.rolling(window, min_periods=1), stat)()
                    )
                else:
                    result[col_name] = getattr(
                        result[col].rolling(window, min_periods=1), stat
                    )()

    logger.debug(f"Added rolling stats for {len(target_cols)} columns")
    return result


def add_holiday_features(
    df: pd.DataFrame,
    date_col: str = "week_end",
    country: str = "CH",
) -> pd.DataFrame:
    """
    Add Swiss holiday indicators.

    Since we have weekly data, we check if any holiday falls in the week.

    Args:
        df: DataFrame with date column
        date_col: Date column name
        country: Country code for holidays

    Returns:
        DataFrame with holiday indicators
    """
    result = df.copy()
    dates = pd.to_datetime(result[date_col])

    # Major Swiss holidays (fixed dates)
    # We'll check if the week contains these dates
    holidays = {
        "new_year": (1, 1),
        "berchtold": (1, 2),
        "easter_monday": None,  # Variable - would need computation
        "labour_day": (5, 1),
        "ascension": None,  # Variable (39 days after Easter)
        "whit_monday": None,  # Variable (50 days after Easter)
        "national_day": (8, 1),
        "christmas": (12, 25),
        "st_stephen": (12, 26),
    }

    # For simplicity, add fixed-date holiday week indicators
    # Filter out variable holidays (None values)
    fixed_holidays = {k: v for k, v in holidays.items() if v is not None}
    for name, (month, day) in fixed_holidays.items():
            # Check if holiday falls in this week
            # Week runs Monday-Sunday, week_end is Sunday
            week_start = dates - pd.Timedelta(days=6)

            # Create holiday dates for each year in data
            years = dates.dt.year.unique()
            holiday_dates = []
            for year in years:
                try:
                    hd = pd.Timestamp(year=year, month=month, day=day)
                    holiday_dates.append(hd)
                except ValueError:
                    pass

            holiday_dates = pd.DatetimeIndex(holiday_dates)

            # Check if holiday in week
            in_week = False
            for hd in holiday_dates:
                in_week = in_week | ((week_start <= hd) & (hd <= dates))

            result[f"holiday_{name}"] = in_week.astype(int)

    # Add generic holiday flag (any holiday in week)
    holiday_cols = [c for c in result.columns if c.startswith("holiday_")]
    if holiday_cols:
        result["holiday_any"] = result[holiday_cols].max(axis=1)

    logger.debug("Added holiday indicators")
    return result


def add_trend_features(
    df: pd.DataFrame,
    date_col: str = "week_end",
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Add linear and quadratic trend features.

    Args:
        df: DataFrame with date column
        date_col: Date column name
        group_col: Grouping column

    Returns:
        DataFrame with trend features
    """
    result = df.copy()
    dates = pd.to_datetime(result[date_col])

    # Continuous time in weeks since first date
    t0 = dates.min()
    weeks_since_start = (dates - t0).dt.days / 7.0

    if group_col:
        # Per-group trends
        for name, group in result.groupby(group_col):
            g_dates = pd.to_datetime(group[date_col])
            g_t0 = g_dates.min()
            g_weeks = (g_dates - g_t0).dt.days / 7.0

            mask = result.index.isin(group.index)
            result.loc[mask, "trend_linear"] = g_weeks
            result.loc[mask, "trend_quadratic"] = g_weeks ** 2
    else:
        result["trend_linear"] = weeks_since_start
        result["trend_quadratic"] = weeks_since_start ** 2

    return result


def add_temporal_features(
    df: pd.DataFrame,
    date_col: str = "week_end",
    target_cols: Optional[List[str]] = None,
    group_col: Optional[str] = None,
    fourier_terms: int = 4,
    lags: List[int] = [1, 2, 3, 4, 8, 12, 26, 52],
    rolling_windows: List[int] = [2, 4, 8, 12, 26],
    add_holidays: bool = True,
    add_trends: bool = True,
) -> pd.DataFrame:
    """
    Add all temporal features in one call.

    This is the main temporal feature engineering function.

    Args:
        df: Input DataFrame
        date_col: Date column name
        target_cols: Target columns for lags/rolling stats
        group_col: Grouping column
        fourier_terms: Number of Fourier term pairs
        lags: Lag periods
        rolling_windows: Rolling window sizes
        add_holidays: Whether to add holiday features
        add_trends: Whether to add trend features

    Returns:
        DataFrame with all temporal features
    """
    logger.info(f"Computing temporal features for {len(df)} records")

    result = df.copy()

    # 1. Fourier seasonal terms
    config = get_config()
    n_terms = fourier_terms or config.features.seasonal.fourier_terms
    result = add_seasonal_fourier(result, date_col, n_terms=n_terms)

    # 2. Multiple seasonalities
    result = add_multiple_seasonalities(result, date_col)

    # 3. Week indicators
    result = add_week_indicators(result, date_col)

    # 4. Lag features
    if target_cols:
        result = add_lag_features(
            result, target_cols, lags=lags, group_col=group_col
        )

    # 5. Rolling statistics
    if target_cols:
        result = add_rolling_stats(
            result, target_cols, windows=rolling_windows, group_col=group_col
        )

    # 6. Holidays
    if add_holidays:
        result = add_holiday_features(result, date_col)

    # 7. Trends
    if add_trends:
        result = add_trend_features(result, date_col, group_col)

    logger.info(f"Temporal features computed: {len(result.columns)} columns")
    return result


# Convenience functions
def get_fourier_features(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Add only Fourier seasonal features."""
    return add_seasonal_fourier(df, **kwargs)


def get_lag_features(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Add only lag features."""
    return add_lag_features(df, **kwargs)


def get_rolling_features(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Add only rolling statistics."""
    return add_rolling_stats(df, **kwargs)


def get_holiday_features(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Add only holiday features."""
    return add_holiday_features(df, **kwargs)