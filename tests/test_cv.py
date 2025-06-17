"""
Tests for rolling origin cross-validation module.
"""
import numpy as np
import pandas as pd
import pytest

from hydro_forecast_pt.evaluation.cv import (
    RollingOriginCV,
    GroupTimeSeriesSplit,
    MultiHorizonRollingCV,
)


class TestRollingOriginCV:
    """Tests for RollingOriginCV class."""

    def setup_method(self):
        """Initialize CV with standard params."""
        self.cv = RollingOriginCV(
            initial_train_weeks=104,
            step_weeks=2,
            test_weeks=2,
            gap_weeks=1,
            group_col="basin_id",
            date_col="week_end",
        )

    def create_sample_data(self, n_weeks: int = 200) -> pd.DataFrame:
        """Create sample data with weekly dates and basins."""
        dates = pd.date_range("2020-01-01", periods=n_weeks, freq="W")
        basins = ["Rhine", "Rhone", "Inn", "Ticino"]

        rows = []
        for basin in basins:
            for d in dates:
                rows.append({
                    "week_end": d,
                    "basin_id": basin,
                    "feature1": np.random.randn(),
                    "feature2": np.random.randn(),
                })

        return pd.DataFrame(rows)

    def test_basic_split(self):
        """Test basic split generation."""
        X = self.create_sample_data(150)

        splits = list(self.cv.split(X))

        # Should have several splits
        assert len(splits) >= 3

        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            # No overlap
            assert len(set(train_idx) & set(test_idx)) == 0

    def test_temporal_ordering(self):
        """Test that train data is always before test data."""
        X = self.create_sample_data(150)

        for train_idx, test_idx in self.cv.split(X):
            train_dates = X.iloc[train_idx]["week_end"]
            test_dates = X.iloc[test_idx]["week_end"]

            assert train_dates.max() < test_dates.min()

    def test_gap_between_train_test(self):
        """Test that there's a gap between train and test."""
        X = self.create_sample_data(150)

        for train_idx, test_idx in self.cv.split(X):
            train_dates = X.iloc[train_idx]["week_end"]
            test_dates = X.iloc[test_idx]["week_end"]

            # Gap should be at least 1 week
            gap = (test_dates.min() - train_dates.max()).days
            assert gap >= 7  # At least 1 week gap

    def test_expanding_window(self):
        """Test that training window expands."""
        X = self.create_sample_data(150)

        train_sizes = []
        for train_idx, test_idx in self.cv.split(X):
            train_sizes.append(len(train_idx))

        # Training size should increase (expanding window)
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i-1]

    def test_group_column_respected(self):
        """Test that group column is used correctly."""
        X = self.create_sample_data(150)

        # All basins should appear in both train and test (per basin evaluation)
        for train_idx, test_idx in self.cv.split(X):
            train_basins = set(X.iloc[train_idx]["basin_id"])
            test_basins = set(X.iloc[test_idx]["basin_id"])

            # In standard rolling origin, all basins in both
            # GroupTimeSeriesSplit would separate them
            assert len(train_basins) == 4
            assert len(test_basins) > 0


class TestGroupTimeSeriesSplit:
    """Tests for GroupTimeSeriesSplit class."""

    def test_basic_split(self):
        """Test basic group-aware split."""
        n_samples = 100
        groups = np.array(["A"] * 25 + ["B"] * 25 + ["C"] * 25 + ["D"] * 25)
        X = pd.DataFrame({"feature": range(n_samples)})

        splitter = GroupTimeSeriesSplit(n_splits=3)

        splits = list(splitter.split(X, groups=groups))

        assert len(splits) >= 1

        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0

            # No group in both train and test
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            assert len(train_groups & test_groups) == 0


class TestMultiHorizonRollingCV:
    """Tests for MultiHorizonRollingCV class."""

    def setup_method(self):
        """Initialize multi-horizon CV."""
        self.cv = MultiHorizonRollingCV(
            horizons=[1, 2, 3, 4, 5, 6],
            initial_train_weeks=104,
            step_weeks=2,
            test_weeks=2,
            gap_weeks=1,
        )

    def create_sample_data(self, n_weeks: int = 200) -> pd.DataFrame:
        """Create sample data."""
        dates = pd.date_range("2020-01-01", periods=n_weeks, freq="W")
        basins = ["Rhine", "Rhone", "Inn", "Ticino"]

        rows = []
        for basin in basins:
            for d in dates:
                rows.append({
                    "week_end": d,
                    "basin_id": basin,
                })

        return pd.DataFrame(rows)

    def test_multi_horizon_splits(self):
        """Test that splits include horizon information."""
        X = self.create_sample_data(150)

        splits = list(self.cv.split(X))

        # Should have splits for each horizon
        assert len(splits) > 6  # 6 horizons * at least 1 split

        # Each split should have 3 elements: train_idx, test_idx, horizon
        for split in splits:
            assert len(split) == 3
            train_idx, test_idx, horizon = split
            assert horizon in [1, 2, 3, 4, 5, 6]

    def test_horizon_coverage(self):
        """Test that all horizons are covered."""
        X = self.create_sample_data(150)

        splits = list(self.cv.split(X))
        horizons = [s[2] for s in splits]

        for h in [1, 2, 3, 4, 5, 6]:
            assert h in horizons


class TestCVEdgeCases:
    """Edge case tests."""

    def test_insufficient_data(self):
        """Test with insufficient data for even one split."""
        X = pd.DataFrame({
            "week_end": pd.date_range("2020-01-01", periods=50, freq="W"),
            "basin_id": ["Rhine"] * 50,
        })

        cv = RollingOriginCV(
            initial_train_weeks=104,
            step_weeks=2,
            test_weeks=2,
            gap_weeks=1,
        )

        splits = list(cv.split(X))
        # Should have at most 1 split or none
        assert len(splits) <= 1

    def test_n_splits_limit(self):
        """Test n_splits parameter limits splits."""
        X = self.create_sample_data(300)

        cv = RollingOriginCV(
            initial_train_weeks=104,
            step_weeks=2,
            test_weeks=2,
            gap_weeks=1,
            n_splits=3,
        )

        splits = list(cv.split(X))
        assert len(splits) <= 3


# Helper for edge case tests
def create_sample_data(n_weeks: int = 200) -> pd.DataFrame:
    """Create sample data with weekly dates and basins."""
    dates = pd.date_range("2020-01-01", periods=n_weeks, freq="W")
    basins = ["Rhine", "Rhone", "Inn", "Ticino"]

    rows = []
    for basin in basins:
        for d in dates:
            rows.append({
                "week_end": d,
                "basin_id": basin,
                "feature1": np.random.randn(),
                "feature2": np.random.randn(),
            })

    return pd.DataFrame(rows)

# Make it available to TestCVEdgeCases
TestCVEdgeCases.create_sample_data = staticmethod(create_sample_data)