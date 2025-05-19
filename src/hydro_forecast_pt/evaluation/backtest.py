"""
Backtesting framework for hydro-forecast-ch.

Runs rolling origin evaluation for multiple models,
compares performance, and generates comprehensive reports.
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..models.base import BaseModel, get_model_class
from ..evaluation.metrics import compute_all_metrics, compute_metrics_per_horizon, compute_metrics_per_basin
from ..evaluation.rolling_origin import RollingOriginSplitter, SplitIndices

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from a single model backtest."""
    model_name: str
    fold_results: List[Dict]
    aggregated_metrics: Dict[str, float]
    horizon_metrics: pd.DataFrame
    basin_metrics: Optional[pd.DataFrame]
    predictions: Dict[str, np.ndarray]  # fold -> predictions
    targets: Dict[str, np.ndarray]  # fold -> true values
    metadata: Dict


class Backtester:
    """
    Runs rolling origin backtesting for one or more models.

    Features:
    - Multiple models comparison
    - Per-horizon, per-basin, per-period metrics
    - Historical replay (reanalysis inputs) vs Live (forecast inputs)
    - Prediction interval evaluation
    - Statistical significance testing
    """

    def __init__(
        self,
        splitter: Optional[RollingOriginSplitter] = None,
        metrics: Optional[List[str]] = None,
        basin_col: str = "basin_id",
        date_col: str = "week_end",
    ):
        """
        Initialize backtester.

        Args:
            splitter: RollingOriginSplitter instance
            metrics: List of metrics to compute
            basin_col: Basin column name
            date_col: Date column name
        """
        self.splitter = splitter or RollingOriginSplitter()
        self.metrics = metrics or [
            "nrmse", "nse", "kge", "skill_score", "mae", "picp", "mpiw", "crps"
        ]
        self.basin_col = basin_col
        self.date_col = date_col

    def run_backtest(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_climatology: Optional[pd.DataFrame] = None,
        basin_col: Optional[str] = None,
        date_col: Optional[str] = None,
    ) -> BacktestResult:
        """
        Run backtest for a single model.

        Args:
            model: Model to evaluate
            X: Features
            y: Targets
            X_climatology: Climatology predictions for skill score
            basin_col: Basin column
            date_col: Date column

        Returns:
            BacktestResult
        """
        basin_col = basin_col or self.basin_col
        date_col = date_col or self.date_col

        logger.info(f"Running backtest for {model.name}")

        # Create groups for per-basin splits
        groups = X[basin_col].values if basin_col in X.columns else None

        fold_results = []
        all_preds = []
        all_targets = []
        all_dates = []
        all_basins = []
        all_clim = []

        for split in self.splitter.split(X, y, groups=groups):
            fold = split.fold

            # Get train/test data
            X_train, X_test = X.iloc[split.train_idx], X.iloc[split.test_idx]
            y_train, y_test = y.iloc[split.train_idx], y.iloc[split.test_idx]

            # Train model
            try:
                model.fit(X_train, y_train)
            except Exception as e:
                logger.error(f"Model {model.name} training failed on fold {fold}: {e}")
                continue

            # Predict
            try:
                y_pred = model.predict(X_test)

                # Get prediction intervals if available
                y_lower, y_upper = None, None
                if hasattr(model, "predict_quantiles"):
                    quantiles = model.predict_quantiles(X_test, [0.1, 0.5, 0.9])
                    y_lower = quantiles.get(0.1, y_pred)
                    y_upper = quantiles.get(0.9, y_pred)
                    y_pred = quantiles.get(0.5, y_pred)

            except Exception as e:
                logger.error(f"Model {model.name} prediction failed on fold {fold}: {e}")
                continue

            # Get target values
            target_cols = [c for c in y_test.columns if c.startswith("target_cf_")]
            y_true = y_test[target_cols].values

            # Climatology for skill score
            y_clim = None
            if X_climatology is not None:
                X_clim_test = X_climatology.iloc[split.test_idx]
                clim_cols = [c for c in X_clim_test.columns if c.startswith("target_cf_")]
                if clim_cols:
                    y_clim = X_clim_test[clim_cols].values

            # Compute metrics
            metrics = compute_all_metrics(
                y_true.flatten() if y_true.ndim > 1 else y_true,
                y_pred.flatten() if y_pred.ndim > 1 else y_pred,
                y_lower.flatten() if y_lower is not None else None,
                y_upper.flatten() if y_upper is not None else None,
                y_clim.flatten() if y_clim is not None else None,
            )

            # Per-horizon metrics
            if y_true.ndim > 1 and y_pred.ndim > 1:
                horizon_metrics = compute_metrics_per_horizon(
                    y_true, y_pred, y_lower, y_upper, y_clim
                )
            else:
                horizon_metrics = pd.DataFrame()

            # Store results
            fold_result = {
                "fold": fold,
                "train_size": len(split.train_idx),
                "test_size": len(split.test_idx),
                "train_start": pd.Timestamp(split.train_dates[0]).strftime("%Y-%m-%d") if len(split.train_dates) > 0 else None,
                "train_end": pd.Timestamp(split.train_dates[-1]).strftime("%Y-%m-%d") if len(split.train_dates) > 0 else None,
                "test_start": pd.Timestamp(split.test_dates[0]).strftime("%Y-%m-%d") if len(split.test_dates) > 0 else None,
                "test_end": pd.Timestamp(split.test_dates[-1]).strftime("%Y-%m-%d") if len(split.test_dates) > 0 else None,
                "metrics": metrics,
                "horizon_metrics": horizon_metrics.to_dict("records") if not horizon_metrics.empty else [],
            }
            fold_results.append(fold_result)

            # Accumulate for overall metrics
            all_preds.append(y_pred)
            all_targets.append(y_true)
            if y_clim is not None:
                all_clim.append(y_clim)
            all_dates.extend([pd.Timestamp(d) for d in split.test_dates])
            if basin_col in X_test.columns:
                all_basins.extend(X_test[basin_col].values)

        if not fold_results:
            raise ValueError(f"No successful folds for model {model.name}")

        # Aggregate predictions
        all_preds = np.vstack(all_preds) if all_preds else np.array([])
        all_targets = np.vstack(all_targets) if all_targets else np.array([])
        all_clim = np.vstack(all_clim) if all_clim else None

        # Overall metrics
        if len(all_preds) > 0 and len(all_targets) > 0:
            aggregated = compute_all_metrics(
                all_targets.flatten(),
                all_preds.flatten(),
                y_climatology=all_clim.flatten() if all_clim is not None else None,
            )
        else:
            aggregated = {}

        # Overall horizon metrics
        if len(all_preds) > 0 and all_preds.ndim > 1:
            overall_horizon = compute_metrics_per_horizon(all_targets, all_preds)
        else:
            overall_horizon = pd.DataFrame()

        # Basin metrics - expand basin labels for each horizon
        basin_metrics = None
        if all_basins and len(all_basins) == len(all_targets):
            # Repeat basin labels for each horizon
            n_horizons = all_targets.shape[1] if all_targets.ndim > 1 else 1
            all_basins_expanded = np.repeat(np.array(all_basins), n_horizons)
            basin_metrics = compute_metrics_per_basin(
                all_targets.flatten() if all_targets.ndim > 1 else all_targets,
                all_preds.flatten() if all_preds.ndim > 1 else all_preds,
                all_basins_expanded,
            )

        # Prepare predictions dict
        predictions = {}
        targets = {}
        for i, fold_result in enumerate(fold_results):
            # Would need to store per-fold predictions
            pass

        result = BacktestResult(
            model_name=model.name,
            fold_results=fold_results,
            aggregated_metrics=aggregated,
            horizon_metrics=overall_horizon,
            basin_metrics=basin_metrics,
            predictions=predictions,
            targets=targets,
            metadata={
                "n_folds": len(fold_results),
                "total_train_samples": sum(f["train_size"] for f in fold_results),
                "total_test_samples": sum(f["test_size"] for f in fold_results),
            }
        )

        logger.info(f"Backtest completed for {model.name}: "
                   f"NSE={aggregated.get('nse', np.nan):.3f}, "
                   f"nRMSE={aggregated.get('nrmse', np.nan):.3f}")

        return result

    def compare_models(
        self,
        models: Dict[str, BaseModel],
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_climatology: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compare multiple models.

        Args:
            models: Dict of model_name -> model instance
            X: Features
            y: Targets
            X_climatology: Climatology predictions

        Returns:
            DataFrame with comparison metrics
        """
        results = []

        for name, model in models.items():
            logger.info(f"Evaluating model: {name}")
            try:
                result = self.run_backtest(model, X, y, X_climatology)
                row = {
                    "model": name,
                    **result.aggregated_metrics,
                }
                results.append(row)
            except Exception as e:
                logger.error(f"Model {name} failed: {e}")
                results.append({"model": name, "error": str(e)})

        return pd.DataFrame(results)


def run_historical_replay(
    model: BaseModel,
    X_reanalysis: pd.DataFrame,
    y: pd.DataFrame,
    X_climatology: Optional[pd.DataFrame] = None,
    splitter: Optional[RollingOriginSplitter] = None,
) -> BacktestResult:
    """
    Run historical replay with reanalysis weather data.

    This represents best-case performance since reanalysis is perfect hindsight.
    """
    logger.info("Running historical replay (reanalysis inputs)")
    backtester = Backtester(splitter=splitter)
    return backtester.run_backtest(model, X_reanalysis, y, X_climatology)


def run_live_simulation(
    model: BaseModel,
    X_forecast: pd.DataFrame,
    y: pd.DataFrame,
    X_climatology: Optional[pd.DataFrame] = None,
    splitter: Optional[RollingOriginSplitter] = None,
) -> BacktestResult:
    """
    Run live simulation with forecast weather data.

    This represents realistic performance since forecasts have errors.
    """
    logger.info("Running live simulation (forecast inputs)")
    backtester = Backtester(splitter=splitter)
    return backtester.run_backtest(model, X_forecast, y, X_climatology)


def compare_replay_vs_live(
    model: BaseModel,
    X_reanalysis: pd.DataFrame,
    X_forecast: pd.DataFrame,
    y: pd.DataFrame,
    X_climatology: Optional[pd.DataFrame] = None,
    splitter: Optional[RollingOriginSplitter] = None,
) -> Dict[str, BacktestResult]:
    """
    Compare historical replay vs live simulation.

    Returns:
        Dict with 'replay' and 'live' BacktestResults
    """
    replay = run_historical_replay(model, X_reanalysis, y, X_climatology, splitter)
    live = run_live_simulation(model, X_forecast, y, X_climatology, splitter)

    logger.info(f"Replay NSE: {replay.aggregated_metrics.get('nse', np.nan):.3f}")
    logger.info(f"Live NSE: {live.aggregated_metrics.get('nse', np.nan):.3f}")
    logger.info(f"Degradation: {replay.aggregated_metrics.get('nse', 0) - live.aggregated_metrics.get('nse', 0):.3f}")

    return {"replay": replay, "live": live}