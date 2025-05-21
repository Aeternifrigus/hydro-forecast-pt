"""
Rolling origin evaluation for hydro-forecast-ch.

Implements time series cross-validation with rolling origin:
- Expanding window (growing training set)
- Sliding window (fixed training set size)
- Configurable gap between train and test
- Horizon-aware splits
"""
import logging
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SplitIndices:
    """Container for train/test split indices."""
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_dates: np.ndarray
    test_dates: np.ndarray
    fold: int
    horizon: int


class RollingOriginSplitter:
    """
    Rolling origin cross-validator for time series.

    Supports:
    - Expanding window: training set grows with each fold
    - Sliding window: training set fixed size, slides forward
    - Multiple horizons: test at different forecast horizons
    - Gap between train and test to avoid leakage
    - Per-basin or global splits
    """

    def __init__(
        self,
        initial_train_size: Union[int, float] = 104,  # weeks (2 years)
        step_size: Union[int, float] = 13,  # weeks (quarterly)
        max_folds: int = 5,
        gap: int = 2,  # weeks
        horizons: List[int] = None,
        expanding: bool = True,
        min_train_size: Optional[int] = None,
        date_col: str = "week_end",
        basin_col: str = "basin_id",
    ):
        """
        Initialize rolling origin splitter.

        Args:
            initial_train_size: Initial training size (weeks or fraction)
            step_size: Step size between folds (weeks or fraction)
            max_folds: Maximum number of folds
            gap: Gap between train and test (weeks)
            horizons: Forecast horizons to evaluate
            expanding: If True, expanding window; if False, sliding window
            min_train_size: Minimum training size (for sliding window)
            date_col: Date column name
            basin_col: Basin column name
        """
        self.initial_train_size = initial_train_size
        self.step_size = step_size
        self.max_folds = max_folds
        self.gap = gap
        self.horizons = horizons or [1, 2, 3, 4, 5, 6]
        self.expanding = expanding
        self.min_train_size = min_train_size
        self.date_col = date_col
        self.basin_col = basin_col

        logger.info(f"RollingOriginSplitter: initial_train={initial_train_size}, "
                   f"step={step_size}, max_folds={max_folds}, gap={gap}, "
                   f"expanding={expanding}")

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.DataFrame] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Generator[SplitIndices, None, None]:
        """
        Generate train/test splits.

        Args:
            X: Features DataFrame
            y: Targets DataFrame (optional)
            groups: Group labels (optional, for per-basin splits)

        Yields:
            SplitIndices for each fold
        """
        if groups is not None:
            # Per-basin splits
            yield from self._split_per_basin(X, y, groups)
        else:
            # Global splits
            yield from self._split_global(X, y)

    def _split_global(self, X: pd.DataFrame, y: Optional[pd.DataFrame]) -> Generator[SplitIndices, None, None]:
        """Global time series splits."""
        n = len(X)

        # Determine initial train size
        if isinstance(self.initial_train_size, float):
            initial_train = int(n * self.initial_train_size)
        else:
            initial_train = min(self.initial_train_size, n - max(self.horizons) - self.gap)

        if isinstance(self.step_size, float):
            step = int(n * self.step_size)
        else:
            step = self.step_size

        # Determine max possible folds
        max_possible_folds = (n - initial_train - self.gap) // step
        n_folds = min(self.max_folds, max_possible_folds)

        if n_folds <= 0:
            logger.warning("Not enough data for rolling origin splits")
            return

        dates = pd.to_datetime(X[self.date_col]).values if self.date_col in X.columns else np.arange(n)

        for fold in range(n_folds):
            if self.expanding:
                train_start = 0
            else:
                # Sliding window
                train_start = fold * step
                if self.min_train_size:
                    train_start = max(0, train_start - (initial_train - self.min_train_size))

            train_end = initial_train + fold * step
            test_start = train_end + self.gap
            test_end = min(test_start + max(self.horizons), n)

            if test_end <= test_start:
                break

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)

            yield SplitIndices(
                train_idx=train_idx,
                test_idx=test_idx,
                train_dates=dates[train_idx],
                test_dates=dates[test_idx],
                fold=fold,
                horizon=max(self.horizons),
            )

    def _split_per_basin(
        self,
        X: pd.DataFrame,
        y: Optional[pd.DataFrame],
        groups: np.ndarray,
    ) -> Generator[SplitIndices, None, None]:
        """Per-basin time series splits."""
        unique_basins = np.unique(groups)

        for basin in unique_basins:
            basin_mask = groups == basin
            basin_X = X[basin_mask].reset_index(drop=True)
            basin_y = y[basin_mask].reset_index(drop=True) if y is not None else None

            n = len(basin_X)
            if n < self.initial_train_size + max(self.horizons) + self.gap:
                logger.warning(f"Basin {basin}: insufficient data for splits")
                continue

            # Use global splitter logic on basin data
            for split in self._split_global(basin_X, basin_y):
                # Map back to original indices
                orig_indices = np.where(basin_mask)[0]
                yield SplitIndices(
                    train_idx=orig_indices[split.train_idx],
                    test_idx=orig_indices[split.test_idx],
                    train_dates=split.train_dates,
                    test_dates=split.test_dates,
                    fold=split.fold,
                    horizon=split.horizon,
                )

    def get_n_splits(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> int:
        """Get number of splits."""
        n = len(X)
        if isinstance(self.initial_train_size, float):
            initial_train = int(n * self.initial_train_size)
        else:
            initial_train = self.initial_train_size

        if isinstance(self.step_size, float):
            step = int(n * self.step_size)
        else:
            step = self.step_size

        max_possible = (n - initial_train - self.gap) // step
        return min(self.max_folds, max(0, max_possible))

    def get_split_dates(self, X: pd.DataFrame) -> List[Dict]:
        """Get split date ranges for reporting."""
        splits = list(self.split(X))
        return [
            {
                "fold": s.fold,
                "train_start": pd.Timestamp(s.train_dates[0]).strftime("%Y-%m-%d") if len(s.train_dates) > 0 else None,
                "train_end": pd.Timestamp(s.train_dates[-1]).strftime("%Y-%m-%d") if len(s.train_dates) > 0 else None,
                "test_start": pd.Timestamp(s.test_dates[0]).strftime("%Y-%m-%d") if len(s.test_dates) > 0 else None,
                "test_end": pd.Timestamp(s.test_dates[-1]).strftime("%Y-%m-%d") if len(s.test_dates) > 0 else None,
                "train_size": len(s.train_idx),
                "test_size": len(s.test_idx),
            }
            for s in splits
        ]


class TimeSeriesSplitter:
    """
    Simplified time series splitter compatible with sklearn interface.
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: Union[int, float] = 13,
        gap: int = 2,
        max_train_size: Optional[int] = None,
    ):
        """
        Initialize.

        Args:
            n_splits: Number of splits
            test_size: Size of test set (weeks or fraction)
            gap: Gap between train and test
            max_train_size: Maximum training set size
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.max_train_size = max_train_size

    def split(self, X, y=None, groups=None):
        """Generate splits."""
        n = len(X)

        if isinstance(self.test_size, float):
            test_n = int(n * self.test_size)
        else:
            test_n = self.test_size

        # Calculate splits from end backwards
        for i in range(self.n_splits):
            test_end = n - i * test_n
            test_start = test_end - test_n
            train_end = test_start - self.gap

            if self.max_train_size:
                train_start = max(0, train_end - self.max_train_size)
            else:
                train_start = 0

            if train_end <= train_start or test_start < train_end:
                break

            yield (
                np.arange(train_start, train_end),
                np.arange(test_start, test_end)
            )

    def get_n_splits(self, X, y=None, groups=None):
        return self.n_splits


def create_cv_splitter(
    strategy: str = "rolling_origin",
    **kwargs,
) -> Union[RollingOriginSplitter, TimeSeriesSplitter]:
    """
    Factory function to create CV splitter.

    Args:
        strategy: 'rolling_origin' or 'time_series'
        **kwargs: Splitter parameters

    Returns:
        CV splitter instance
    """
    if strategy == "rolling_origin":
        return RollingOriginSplitter(**kwargs)
    elif strategy == "time_series":
        return TimeSeriesSplitter(**kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")