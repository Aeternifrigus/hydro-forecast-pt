"""
Evaluation metrics for hydro-forecast-ch.

Implements standard hydrological and forecasting metrics:
- nRMSE (normalized Root Mean Square Error)
- NSE (Nash-Sutcliffe Efficiency)
- KGE (Kling-Gupta Efficiency)
- Skill Score
- Prediction Interval Coverage Probability (PICP)
- Mean Prediction Interval Width (MPIW)
- CRPS (Continuous Ranked Probability Score)
- Quantile Loss (Pinball Loss)
"""
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def nrmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize_by: str = "mean",
) -> float:
    """
    Normalized Root Mean Square Error.

    nRMSE = RMSE / normalizer

    Args:
        y_true: True values
        y_pred: Predicted values
        normalize_by: Normalization method ('mean', 'std', 'range', 'iqr')

    Returns:
        nRMSE value
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not np.any(mask):
        return np.nan

    y_t = y_true[mask]
    y_p = y_pred[mask]

    rmse = np.sqrt(np.mean((y_t - y_p) ** 2))

    if normalize_by == "mean":
        normalizer = np.mean(np.abs(y_t))
    elif normalize_by == "std":
        normalizer = np.std(y_t)
    elif normalize_by == "range":
        normalizer = np.max(y_t) - np.min(y_t)
    elif normalize_by == "iqr":
        normalizer = np.percentile(y_t, 75) - np.percentile(y_t, 25)
    else:
        normalizer = np.mean(np.abs(y_t))

    if normalizer == 0:
        return np.nan

    return rmse / normalizer


def nse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Nash-Sutcliffe Efficiency.

    NSE = 1 - Σ(y - ŷ)² / Σ(y - ȳ)²

    Range: (-∞, 1], 1 = perfect, 0 = mean predictor, <0 = worse than mean

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        NSE value
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not np.any(mask):
        return np.nan

    y_t = y_true[mask]
    y_p = y_pred[mask]

    numerator = np.sum((y_t - y_p) ** 2)
    denominator = np.sum((y_t - np.mean(y_t)) ** 2)

    if denominator == 0:
        return np.nan

    return 1 - numerator / denominator


def kge(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Kling-Gupta Efficiency.

    KGE = 1 - √[(r-1)² + (α-1)² + (β-1)²]

    Where:
    - r = correlation coefficient
    - α = σ_pred / σ_true (variability ratio)
    - β = μ_pred / μ_true (bias ratio)

    Range: (-∞, 1], 1 = perfect

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        KGE value
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not np.any(mask):
        return np.nan

    y_t = y_true[mask]
    y_p = y_pred[mask]

    # Correlation
    r = np.corrcoef(y_t, y_p)[0, 1]
    if np.isnan(r):
        r = 0

    # Variability ratio
    alpha = np.std(y_p) / np.std(y_t) if np.std(y_t) > 0 else 1

    # Bias ratio
    beta = np.mean(y_p) / np.mean(y_t) if np.mean(y_t) > 0 else 1

    kge_val = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return kge_val


def skill_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_climatology: np.ndarray,
) -> float:
    """
    Skill score relative to climatology.

    Skill = 1 - MSE_model / MSE_climatology

    > 0 means better than climatology
    < 0 means worse than climatology

    Args:
        y_true: True values
        y_pred: Model predictions
        y_climatology: Climatology predictions

    Returns:
        Skill score
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(y_climatology))
    if not np.any(mask):
        return np.nan

    y_t = y_true[mask]
    y_p = y_pred[mask]
    y_c = y_climatology[mask]

    mse_model = np.mean((y_t - y_p) ** 2)
    mse_clim = np.mean((y_t - y_c) ** 2)

    if mse_clim == 0:
        return np.nan

    return 1 - mse_model / mse_clim


def mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Mean Absolute Error."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs(y_true[mask] - y_pred[mask]))


def mse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Mean Squared Error."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not np.any(mask):
        return np.nan
    return np.mean((y_true[mask] - y_pred[mask]) ** 2)


def rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(mse(y_true, y_pred))


def mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Mean Absolute Percentage Error."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not np.any(mask):
        return np.nan
    y_t = y_true[mask]
    y_p = y_pred[mask]
    # Avoid division by zero
    mask_nz = y_t != 0
    if not np.any(mask_nz):
        return np.nan
    return np.mean(np.abs((y_t[mask_nz] - y_p[mask_nz]) / y_t[mask_nz])) * 100


def pbias(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Percent Bias."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not np.any(mask):
        return np.nan
    y_t = y_true[mask]
    y_p = y_pred[mask]
    return 100 * np.sum(y_p - y_t) / np.sum(y_t)


def correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Pearson correlation coefficient."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not np.any(mask):
        return np.nan
    return np.corrcoef(y_true[mask], y_pred[mask])[0, 1]


# ============================================================================
# Probabilistic Metrics
# ============================================================================

def picp(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> float:
    """
    Prediction Interval Coverage Probability.

    PICP = fraction of true values within [lower, upper]

    Args:
        y_true: True values
        y_lower: Lower bound of prediction interval
        y_upper: Upper bound of prediction interval

    Returns:
        Coverage probability (0-1)
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_lower) | np.isnan(y_upper))
    if not np.any(mask):
        return np.nan

    y_t = y_true[mask]
    y_l = y_lower[mask]
    y_u = y_upper[mask]

    covered = (y_t >= y_l) & (y_t <= y_u)
    return np.mean(covered)


def mpiw(
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> float:
    """
    Mean Prediction Interval Width.

    MPIW = mean(upper - lower)

    Args:
        y_lower: Lower bound
        y_upper: Upper bound

    Returns:
        Mean interval width
    """
    mask = ~(np.isnan(y_lower) | np.isnan(y_upper))
    if not np.any(mask):
        return np.nan
    return np.mean(y_upper[mask] - y_lower[mask])


def nmpiw(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> float:
    """
    Normalized Mean Prediction Interval Width.

    NMPIW = MPIW / (max(y_true) - min(y_true))
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_lower) | np.isnan(y_upper))
    if not np.any(mask):
        return np.nan

    y_t = y_true[mask]
    y_l = y_lower[mask]
    y_u = y_upper[mask]

    mpiw_val = np.mean(y_u - y_l)
    range_val = np.max(y_t) - np.min(y_t)

    if range_val == 0:
        return np.nan

    return mpiw_val / range_val


def winkler_score(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """
    Winkler Score (Interval Score) for prediction intervals.

    The Winkler score rewards narrow intervals that contain the true value,
    and penalizes both wide intervals and observations outside the interval.

    Score = (upper - lower) + (2/alpha) * (lower - y_true) * I(y_true < lower)
                  + (2/alpha) * (y_true - upper) * I(y_true > upper)

    Args:
        y_true: True values
        y_lower: Lower bound of prediction interval
        y_upper: Upper bound of prediction interval
        alpha: Significance level (e.g., 0.1 for 90% interval)

    Returns:
        Mean Winkler score (lower is better)
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_lower) | np.isnan(y_upper))
    if not np.any(mask):
        return np.nan

    y_t = y_true[mask]
    y_l = y_lower[mask]
    y_u = y_upper[mask]

    # Interval width
    width = y_u - y_l

    # Penalty for observations below lower bound
    below = y_t < y_l
    penalty_below = (2 / alpha) * (y_l - y_t) * below

    # Penalty for observations above upper bound
    above = y_t > y_u
    penalty_above = (2 / alpha) * (y_t - y_u) * above

    score = width + penalty_below + penalty_above
    return np.mean(score)


def crps(
    y_true: np.ndarray,
    y_pred_quantiles: np.ndarray,
    quantiles: List[float],
) -> float:
    """
    Continuous Ranked Probability Score.

    Approximates CRPS using quantile predictions.

    Args:
        y_true: True values (n_samples,)
        y_pred_quantiles: Predicted quantiles (n_samples, n_quantiles)
        quantiles: Quantile levels

    Returns:
        Mean CRPS
    """
    if len(y_true) != y_pred_quantiles.shape[0]:
        raise ValueError("Shape mismatch")

    mask = ~np.isnan(y_true)
    if not np.any(mask):
        return np.nan

    y_t = y_true[mask]
    y_q = y_pred_quantiles[mask]

    # CRPS approximation using quantile formula
    # CRPS ≈ (1/K) * Σ |y_t - q_k| * (2q_k - 1) for symmetric quantiles
    # More accurate: use trapezoidal integration

    crps_vals = []
    for i in range(len(y_t)):
        q_vals = y_q[i]
        # Sort by quantiles
        sorted_idx = np.argsort(quantiles)
        q_sorted = np.array(quantiles)[sorted_idx]
        pred_sorted = q_vals[sorted_idx]

        # Trapezoidal integration
        integrand = np.abs(pred_sorted - y_t[i]) * np.diff(np.concatenate([[0], q_sorted, [1]]))
        crps_i = np.sum(integrand)
        crps_vals.append(crps_i)

    return np.mean(crps_vals)


def quantile_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 0.5,
) -> float:
    """
    Pinball loss for quantile regression.

    L_α(y, ŷ) = max(α(y - ŷ), (α - 1)(y - ŷ))

    Args:
        y_true: True values
        y_pred: Predicted quantile
        alpha: Quantile level

    Returns:
        Mean pinball loss
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not np.any(mask):
        return np.nan

    y_t = y_true[mask]
    y_p = y_pred[mask]

    error = y_t - y_p
    loss = np.mean(np.maximum(alpha * error, (alpha - 1) * error))
    return loss


def winker_score(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """
    Winkler score (interval score).

    Penalizes both interval width and lack of coverage.

    IS = (upper - lower) + 2/α * (lower - y) * I(y < lower) + 2/α * (y - upper) * I(y > upper)

    Args:
        y_true: True values
        y_lower: Lower bound
        y_upper: Upper bound
        alpha: Significance level (e.g., 0.1 for 90% interval)

    Returns:
        Mean Winkler score
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_lower) | np.isnan(y_upper))
    if not np.any(mask):
        return np.nan

    y_t = y_true[mask]
    y_l = y_lower[mask]
    y_u = y_upper[mask]

    width = y_u - y_l
    below = np.where(y_t < y_l, (y_l - y_t) * 2 / alpha, 0)
    above = np.where(y_t > y_u, (y_t - y_u) * 2 / alpha, 0)

    return np.mean(width + below + above)


# ============================================================================
# Aggregated Metrics
# ============================================================================

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lower: Optional[np.ndarray] = None,
    y_upper: Optional[np.ndarray] = None,
    y_climatology: Optional[np.ndarray] = None,
    quantiles: Optional[List[float]] = None,
    y_pred_quantiles: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute all available metrics.

    Args:
        y_true: True values
        y_pred: Point predictions (median)
        y_lower: Lower prediction interval
        y_upper: Upper prediction interval
        y_climatology: Climatology predictions
        quantiles: Quantile levels for y_pred_quantiles
        y_pred_quantiles: All quantile predictions (n_samples, n_quantiles)

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Deterministic metrics
    metrics["nrmse"] = nrmse(y_true, y_pred)
    metrics["nse"] = nse(y_true, y_pred)
    metrics["kge"] = kge(y_true, y_pred)
    metrics["mae"] = mae(y_true, y_pred)
    metrics["mse"] = mse(y_true, y_pred)
    metrics["rmse"] = rmse(y_true, y_pred)
    metrics["mape"] = mape(y_true, y_pred)
    metrics["pbias"] = pbias(y_true, y_pred)
    metrics["correlation"] = correlation(y_true, y_pred)

    # Skill score
    if y_climatology is not None:
        metrics["skill_score"] = skill_score(y_true, y_pred, y_climatology)

    # Probabilistic metrics
    if y_lower is not None and y_upper is not None:
        metrics["picp"] = picp(y_true, y_lower, y_upper)
        metrics["mpiw"] = mpiw(y_lower, y_upper)
        metrics["nmpiw"] = nmpiw(y_true, y_lower, y_upper)
        metrics["winkler_score"] = winkler_score(y_true, y_lower, y_upper)

    # CRPS
    if y_pred_quantiles is not None and quantiles is not None:
        metrics["crps"] = crps(y_true, y_pred_quantiles, quantiles)

    return metrics


def compute_metrics_per_horizon(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lower: Optional[np.ndarray] = None,
    y_upper: Optional[np.ndarray] = None,
    y_climatology: Optional[np.ndarray] = None,
    quantiles: Optional[List[float]] = None,
    y_pred_quantiles: Optional[np.ndarray] = None,
    horizon_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute metrics for each horizon separately.

    Args:
        y_true: True values (n_samples, n_horizons)
        y_pred: Predictions (n_samples, n_horizons)
        y_lower: Lower bounds (n_samples, n_horizons)
        y_upper: Upper bounds (n_samples, n_horizons)
        y_climatology: Climatology (n_samples, n_horizons)
        quantiles: Quantile levels
        y_pred_quantiles: Quantile predictions (n_samples, n_horizons, n_quantiles)
        horizon_names: Names for horizons

    Returns:
        DataFrame with metrics per horizon
    """
    n_horizons = y_true.shape[1]
    if horizon_names is None:
        horizon_names = [f"h{h}" for h in range(1, n_horizons + 1)]

    results = []

    for h in range(n_horizons):
        h_metrics = compute_all_metrics(
            y_true[:, h],
            y_pred[:, h],
            y_lower[:, h] if y_lower is not None else None,
            y_upper[:, h] if y_upper is not None else None,
            y_climatology[:, h] if y_climatology is not None else None,
            quantiles,
            y_pred_quantiles[:, h, :] if y_pred_quantiles is not None else None,
        )
        h_metrics["horizon"] = horizon_names[h]
        results.append(h_metrics)

    return pd.DataFrame(results)


def compute_metrics_per_basin(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    basins: np.ndarray,
    y_lower: Optional[np.ndarray] = None,
    y_upper: Optional[np.ndarray] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Compute metrics for each basin separately.

    Args:
        y_true: True values
        y_pred: Predictions
        basins: Basin identifiers
        y_lower: Lower bounds
        y_upper: Upper bounds
        **kwargs: Additional arguments to compute_all_metrics

    Returns:
        DataFrame with metrics per basin
    """
    unique_basins = np.unique(basins)
    results = []

    for basin in unique_basins:
        mask = basins == basin
        if np.sum(mask) < 5:
            continue

        h_metrics = compute_all_metrics(
            y_true[mask],
            y_pred[mask],
            y_lower[mask] if y_lower is not None else None,
            y_upper[mask] if y_upper is not None else None,
            **kwargs,
        )
        h_metrics["basin"] = basin
        results.append(h_metrics)

    return pd.DataFrame(results)


def compute_metrics_per_period(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: np.ndarray,
    y_lower: Optional[np.ndarray] = None,
    y_upper: Optional[np.ndarray] = None,
    period_col: str = "month",
    **kwargs,
) -> pd.DataFrame:
    """
    Compute metrics grouped by time period.

    Args:
        y_true: True values
        y_pred: Predictions
        dates: Date array
        y_lower: Lower bounds
        y_upper: Upper bounds
        period_col: Period to group by ('month', 'season', 'year')
        **kwargs: Additional arguments

    Returns:
        DataFrame with metrics per period
    """
    dates = pd.to_datetime(dates)

    if period_col == "month":
        periods = dates.month
    elif period_col == "season":
        # DJF, MAM, JJA, SON
        month = dates.month
        periods = pd.cut(month, bins=[0, 3, 6, 9, 12], labels=["DJF", "MAM", "JJA", "SON"])
    elif period_col == "year":
        periods = dates.year
    else:
        raise ValueError(f"Unknown period_col: {period_col}")

    unique_periods = np.unique(periods)
    results = []

    for period in unique_periods:
        mask = periods == period
        if np.sum(mask) < 5:
            continue

        h_metrics = compute_all_metrics(
            y_true[mask],
            y_pred[mask],
            y_lower[mask] if y_lower is not None else None,
            y_upper[mask] if y_upper is not None else None,
            **kwargs,
        )
        h_metrics[period_col] = period
        results.append(h_metrics)

    return pd.DataFrame(results)
# 2025-09-04: Add NSE/KGE/nRMSE: Nash-Sutcliffe, Kling-Gupta, normalized RMSE

# 2025-09-06: Improve CV strategy: PIC and width for 0.1/0.9 quantiles

# 2025-09-11: Improve CV strategy: 1-6w + aggregate, per-basin, per-season

# 2025-09-14: Add NSE/KGE/nRMSE: Nash-Sutcliffe, Kling-Gupta, normalized RMSE

# 2025-09-28: Evaluation: PIC and width for 0.1/0.9 quantiles

# 2025-10-09: Improve CV strategy: PIC and width for 0.1/0.9 quantiles

# 2025-10-19: Improve CV strategy: Nash-Sutcliffe, Kling-Gupta, normalized RMSE

# 2025-10-24: Improve CV strategy: PIC and width for 0.1/0.9 quantiles

# 2025-10-25: Improve CV strategy: Nash-Sutcliffe, Kling-Gupta, normalized RMSE

# 2025-10-31: Add prediction intervals: PIC and width for 0.1/0.9 quantiles
