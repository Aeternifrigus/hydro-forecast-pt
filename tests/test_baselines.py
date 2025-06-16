"""
Tests for baseline models.
"""
import numpy as np
import pandas as pd
import pytest

from hydro_forecast_pt.models.baselines import (
    ClimatologyModel,
    PersistenceModel,
    SeasonalPersistenceModel,
    LinearTrendModel,
    NaiveEnsembleModel,
)


class TestClimatologyModel:
    """Tests for ClimatologyModel."""

    def setup_method(self):
        """Create synthetic training data."""
        np.random.seed(42)
        n_samples = 200

        # Create data with basin_id, week, year
        basins = ["rhine", "rhone", "inn", "ticino"]
        data = []

        for basin in basins:
            for year in [2023, 2024]:
                for week in range(1, 53):
                    # Seasonal pattern + noise
                    seasonal = 0.3 + 0.4 * np.sin(2 * np.pi * week / 52)
                    noise = np.random.normal(0, 0.05)
                    cf = np.clip(seasonal + noise, 0, 1)

                    data.append({
                        "basin_id": basin,
                        "year": year,
                        "week": week,
                        "week_end": pd.Timestamp(f"{year}-01-01") + pd.Timedelta(weeks=week),
                        "target_cf_h1w": cf,
                        "target_cf_h2w": cf + np.random.normal(0, 0.02),
                    })

        df = pd.DataFrame(data)
        self.X = df[["basin_id", "year", "week", "week_end"]]
        self.y = df[["target_cf_h1w", "target_cf_h2w"]]

    def test_fit(self):
        """Test that model fits without error."""
        model = ClimatologyModel()
        model.fit(self.X, self.y)
        assert model.is_fitted
        assert len(model.climatology) > 0
        assert model.overall_mean > 0

    def test_predict_shape(self):
        """Test prediction shape matches horizons."""
        model = ClimatologyModel().fit(self.X, self.y)
        preds = model.predict(self.X.iloc[:10])
        assert preds.shape == (10, 2)

    def test_predict_uses_basin_week(self):
        """Test that predictions use basin-specific climatology."""
        model = ClimatologyModel().fit(self.X, self.y)

        # Predict for a specific basin/week
        X_test = pd.DataFrame({
            "basin_id": ["rhine"],
            "year": [2025],
            "week": [1],
            "week_end": [pd.Timestamp("2025-01-07")]
        })
        preds = model.predict(X_test)

        # Should match climatology for rhine week 1
        expected = model.climatology.get(("rhine", 1), model.overall_mean)
        assert np.isclose(preds[0, 0], expected)

    def test_predict_fallback_to_global(self):
        """Test fallback to global climatology for unseen basin."""
        model = ClimatologyModel().fit(self.X, self.y)

        X_test = pd.DataFrame({
            "basin_id": ["unknown_basin"],
            "year": [2025],
            "week": [1],
            "week_end": [pd.Timestamp("2025-01-07")]
        })
        preds = model.predict(X_test)

        # Should use global climatology for week 1
        expected = model.global_climatology.get(1, model.overall_mean)
        assert np.isclose(preds[0, 0], expected)

    def test_missing_week_column_raises(self):
        """Test that missing week column raises error."""
        model = ClimatologyModel()
        X_bad = self.X.drop(columns=["week"])
        with pytest.raises(ValueError, match="week"):
            model.fit(X_bad, self.y)


class TestPersistenceModel:
    """Tests for PersistenceModel."""

    def setup_method(self):
        """Create synthetic training data."""
        np.random.seed(42)
        basins = ["rhine", "rhone"]
        data = []

        for basin in basins:
            for week in range(1, 53):
                cf = 0.4 + 0.1 * np.sin(2 * np.pi * week / 52) + np.random.normal(0, 0.03)
                cf = np.clip(cf, 0, 1)
                data.append({
                    "basin_id": basin,
                    "year": 2024,
                    "week": week,
                    "week_end": pd.Timestamp("2024-01-01") + pd.Timedelta(weeks=week),
                    "target_cf_h1w": cf,
                    "target_cf_h2w": cf,
                })

        df = pd.DataFrame(data)
        self.X = df[["basin_id", "year", "week", "week_end"]]
        self.y = df[["target_cf_h1w", "target_cf_h2w"]]

    def test_fit(self):
        """Test that model fits without error."""
        model = PersistenceModel()
        model.fit(self.X, self.y)
        assert model.is_fitted
        assert len(model.last_values) == 2  # Two basins

    def test_predict_shape(self):
        """Test prediction shape."""
        model = PersistenceModel().fit(self.X, self.y)
        preds = model.predict(self.X.iloc[:5])
        assert preds.shape == (5, 2)

    def test_predict_uses_last_value(self):
        """Test that prediction uses last observed value per basin."""
        model = PersistenceModel().fit(self.X, self.y)

        # For rhine, last week is 52
        X_test = pd.DataFrame({
            "basin_id": ["rhine"],
            "year": [2025],
            "week": [1],
            "week_end": [pd.Timestamp("2025-01-07")]
        })
        preds = model.predict(X_test)

        # Should use last value for rhine (week 52 of 2024)
        expected = model.last_values.get("rhine", model.global_last)
        assert np.isclose(preds[0, 0], expected)


class TestSeasonalPersistenceModel:
    """Tests for SeasonalPersistenceModel."""

    def setup_method(self):
        """Create synthetic training data with two years."""
        np.random.seed(42)
        basins = ["rhine", "rhone"]
        data = []

        for basin in basins:
            for year in [2023, 2024]:
                for week in range(1, 53):
                    cf = 0.4 + 0.3 * np.sin(2 * np.pi * week / 52) + np.random.normal(0, 0.03)
                    cf = np.clip(cf, 0, 1)
                    data.append({
                        "basin_id": basin,
                        "year": year,
                        "week": week,
                        "week_end": pd.Timestamp(f"{year}-01-01") + pd.Timedelta(weeks=week),
                        "target_cf_h1w": cf,
                        "target_cf_h2w": cf,
                    })

        df = pd.DataFrame(data)
        self.X = df[["basin_id", "year", "week", "week_end"]]
        self.y = df[["target_cf_h1w", "target_cf_h2w"]]

    def test_fit(self):
        """Test that model fits without error."""
        model = SeasonalPersistenceModel()
        model.fit(self.X, self.y)
        assert model.is_fitted
        assert len(model.history) == 4 * 52  # 2 basins * 2 years * 52 weeks

    def test_predict_uses_same_week_previous_year(self):
        """Test that prediction uses same week from previous year."""
        model = SeasonalPersistenceModel().fit(self.X, self.y)

        # Predict week 10 of 2025 -> should use week 10 of 2024
        X_test = pd.DataFrame({
            "basin_id": ["rhine"],
            "year": [2025],
            "week": [10],
            "week_end": [pd.Timestamp("2025-03-07")]
        })
        preds = model.predict(X_test)

        # Should use value from (rhine, 2024, 10)
        expected = model.history.get(("rhine", 2024, 10))
        assert np.isclose(preds[0, 0], expected)


class TestLinearTrendModel:
    """Tests for LinearTrendModel."""

    def setup_method(self):
        """Create synthetic training data with trend."""
        np.random.seed(42)
        basins = ["rhine", "rhone"]
        data = []

        for basin in basins:
            for year in [2020, 2021, 2022, 2023, 2024]:
                for week in range(1, 53):
                    # Add a slight trend + seasonal
                    trend = 0.005 * (year - 2020)  # Small trend
                    seasonal = 0.3 + 0.3 * np.sin(2 * np.pi * week / 52)
                    noise = np.random.normal(0, 0.03)
                    cf = np.clip(trend + seasonal + noise, 0, 1)

                    data.append({
                        "basin_id": basin,
                        "year": year,
                        "week": week,
                        "week_end": pd.Timestamp(f"{year}-01-01") + pd.Timedelta(weeks=week),
                        "target_cf_h1w": cf,
                        "target_cf_h2w": cf,
                    })

        df = pd.DataFrame(data)
        self.X = df[["basin_id", "year", "week", "week_end"]]
        self.y = df[["target_cf_h1w", "target_cf_h2w"]]

    def test_fit(self):
        """Test that model fits without error."""
        model = LinearTrendModel()
        model.fit(self.X, self.y)
        assert model.is_fitted
        assert len(model.models) > 0

    def test_predict_shape(self):
        """Test prediction shape."""
        model = LinearTrendModel().fit(self.X, self.y)
        preds = model.predict(self.X.iloc[:10])
        assert preds.shape == (10, 2)


class TestNaiveEnsembleModel:
    """Tests for NaiveEnsembleModel."""

    def setup_method(self):
        """Create synthetic training data."""
        np.random.seed(42)
        basins = ["rhine", "rhone"]
        data = []

        for basin in basins:
            for year in [2023, 2024]:
                for week in range(1, 53):
                    cf = 0.4 + 0.3 * np.sin(2 * np.pi * week / 52) + np.random.normal(0, 0.03)
                    cf = np.clip(cf, 0, 1)
                    data.append({
                        "basin_id": basin,
                        "year": year,
                        "week": week,
                        "week_end": pd.Timestamp(f"{year}-01-01") + pd.Timedelta(weeks=week),
                        "target_cf_h1w": cf,
                        "target_cf_h2w": cf,
                    })

        df = pd.DataFrame(data)
        self.X = df[["basin_id", "year", "week", "week_end"]]
        self.y = df[["target_cf_h1w", "target_cf_h2w"]]

    def test_fit(self):
        """Test that ensemble fits all submodels."""
        model = NaiveEnsembleModel()
        model.fit(self.X, self.y)
        assert model.is_fitted
        assert len(model.submodels) == 3

    def test_predict_shape(self):
        """Test prediction shape."""
        model = NaiveEnsembleModel().fit(self.X, self.y)
        preds = model.predict(self.X.iloc[:10])
        assert preds.shape == (10, 2)

    def test_predict_weights(self):
        """Test that weights are applied correctly."""
        model = NaiveEnsembleModel(
            weights={"climatology": 1.0, "persistence": 0.0, "seasonal_persistence": 0.0}
        )
        model.fit(self.X, self.y)
        preds = model.predict(self.X.iloc[:5])

        # Should match climatology predictions exactly
        clim = ClimatologyModel().fit(self.X, self.y)
        clim_preds = clim.predict(self.X.iloc[:5])
        assert np.allclose(preds, clim_preds)


class TestBaselineEdgeCases:
    """Edge case tests for all baselines."""

    def test_empty_data_raises(self):
        """Test that empty data raises appropriate errors."""
        X = pd.DataFrame(columns=["basin_id", "year", "week", "week_end"])
        y = pd.DataFrame(columns=["target_cf_h1w", "target_cf_h2w"])

        for model_class in [ClimatologyModel, PersistenceModel, SeasonalPersistenceModel]:
            model = model_class()
            with pytest.raises((ValueError, IndexError)):
                model.fit(X, y)

    def test_unfitted_predict_raises(self):
        """Test that predicting before fitting raises error."""
        X = pd.DataFrame({
            "basin_id": ["rhine"],
            "year": [2024],
            "week": [1],
            "week_end": [pd.Timestamp("2024-01-07")]
        })

        for model_class in [ClimatologyModel, PersistenceModel, SeasonalPersistenceModel]:
            model = model_class()
            with pytest.raises(ValueError, match="not fitted"):
                model.predict(X)