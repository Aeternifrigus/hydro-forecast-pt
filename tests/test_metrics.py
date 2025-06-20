"""
Tests for evaluation metrics module.
"""
import numpy as np
import pandas as pd
import pytest

from hydro_forecast_pt.evaluation.metrics import (
    nse,
    kge,
    nrmse,
    skill_score,
    picp,
    mpiw,
    mae,
    mape,
    pbias,
    correlation,
    compute_all_metrics,
)


class TestNSE:
    """Tests for Nash-Sutcliffe Efficiency."""

    def test_nse_perfect(self):
        """Test NSE with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = nse(y_true, y_pred)
        assert np.isclose(result, 1.0)

    def test_nse_mean_predictor(self):
        """Test NSE with mean predictor (should be 0)."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([3.0, 3.0, 3.0, 3.0, 3.0])  # mean = 3

        result = nse(y_true, y_pred)
        assert np.isclose(result, 0.0)

    def test_nse_worse_than_mean(self):
        """Test NSE with predictions worse than mean (negative)."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # Reversed

        result = nse(y_true, y_pred)
        assert result < 0

    def test_nse_nan_handling(self):
        """Test NSE handles NaN."""
        y_true = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = nse(y_true, y_pred)
        # Should compute on non-NaN values
        assert result > 0


class TestKGE:
    """Tests for Kling-Gupta Efficiency."""

    def test_kge_perfect(self):
        """Test KGE with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = kge(y_true, y_pred)
        assert np.isclose(result, 1.0)

    def test_kge_bias(self):
        """Test KGE with biased predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([2.0, 3.0, 4.0, 5.0, 6.0])  # +1 bias

        result = kge(y_true, y_pred)
        assert result < 1.0  # Should be less than perfect


class TestNRMSE:
    """Tests for Normalized RMSE."""

    def test_nrmse_perfect(self):
        """Test nRMSE with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = nrmse(y_true, y_pred)
        assert np.isclose(result, 0.0)

    def test_nrmse_constant(self):
        """Test nRMSE with constant predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([3.0, 3.0, 3.0, 3.0, 3.0])

        result = nrmse(y_true, y_pred, normalize_by="range")
        # RMSE = sqrt(mean((y-3)^2)) = sqrt((4+1+0+1+4)/5) = sqrt(2) ≈ 1.414
        # Range = 5-1 = 4
        # nRMSE = 1.414 / 4 ≈ 0.354
        assert np.isclose(result, np.sqrt(2) / 4, atol=0.01)

    def test_nrmse_flat_true(self):
        """Test nRMSE with flat true values."""
        y_true = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        y_pred = np.array([3.0, 3.0, 3.0, 3.0, 3.0])

        result = nrmse(y_true, y_pred)
        # Should handle zero range
        assert np.isnan(result) or result == 0.0


class TestSkillScore:
    """Tests for Skill Score."""

    def test_skill_perfect(self):
        """Test skill with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_clim = np.array([3.0, 3.0, 3.0, 3.0, 3.0])

        result = skill_score(y_true, y_pred, y_clim)
        assert np.isclose(result, 1.0)

    def test_skill_equal_to_climatology(self):
        """Test skill equal to climatology."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        y_clim = np.array([3.0, 3.0, 3.0, 3.0, 3.0])

        result = skill_score(y_true, y_pred, y_clim)
        assert np.isclose(result, 0.0)

    def test_skill_worse_than_climatology(self):
        """Test skill worse than climatology."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        y_clim = np.array([3.0, 3.0, 3.0, 3.0, 3.0])

        result = skill_score(y_true, y_pred, y_clim)
        assert result < 0.0


class TestPredictionInterval:
    """Tests for prediction interval metrics."""

    def test_picp_perfect(self):
        """Test PICP with perfect coverage."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_lower = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        y_upper = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

        result = picp(y_true, y_lower, y_upper)
        # All 5 points should be in interval
        assert np.isclose(result, 1.0)

    def test_picp_partial(self):
        """Test PICP with partial coverage."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_lower = np.array([1.5, 1.5, 2.5, 3.5, 4.5])
        y_upper = np.array([2.5, 2.5, 3.5, 4.5, 5.5])

        result = picp(y_true, y_lower, y_upper)
        # Only 4 out of 5 in interval
        assert np.isclose(result, 0.8)

    def test_mpiw(self):
        """Test mean interval width."""
        y_lower = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        y_upper = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

        result = mpiw(y_lower, y_upper)
        # All intervals have width 1.0
        assert np.isclose(result, 1.0)


class TestOtherMetrics:
    """Tests for other metrics."""

    def test_mae(self):
        """Test MAE."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.5, 2.5, 2.5, 4.5, 5.5])

        result = mae(y_true, y_pred)
        expected = np.mean([0.5, 0.5, 0.5, 0.5, 0.5])
        assert np.isclose(result, expected)

    def test_mape(self):
        """Test MAPE."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y_pred = np.array([11.0, 19.0, 33.0, 38.0, 55.0])

        result = mape(y_true, y_pred)
        # % errors: 10%, 5%, 10%, 5%, 10% -> mean = 8%
        assert np.isclose(result, 8.0, atol=0.1)

    def test_pbias(self):
        """Test PBIAS."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y_pred = np.array([11.0, 19.0, 33.0, 38.0, 55.0])

        result = pbias(y_true, y_pred)
        # Sum(y_pred - y_true) = 1-1+3-2+5 = 6
        # Sum(y_true) = 150
        # PBIAS = 6/150 * 100 = 4%
        assert np.isclose(result, 4.0, atol=0.1)

    def test_correlation(self):
        """Test correlation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = correlation(y_true, y_pred)
        assert np.isclose(result, 1.0)


class TestComputeAllMetrics:
    """Tests for compute_all_metrics."""

    def test_compute_all_metrics_basic(self):
        """Test compute_all_metrics with basic inputs."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 5.1])

        result = compute_all_metrics(y_true, y_pred)

        assert "nrmse" in result
        assert "nse" in result
        assert "kge" in result
        assert "mae" in result
        assert "mse" in result
        assert "rmse" in result
        assert "mape" in result
        assert "pbias" in result
        assert "correlation" in result

    def test_compute_all_metrics_with_intervals(self):
        """Test compute_all_metrics with prediction intervals."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 5.1])
        y_lower = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        y_upper = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

        result = compute_all_metrics(y_true, y_pred, y_lower, y_upper)

        assert "picp" in result
        assert "mpiw" in result
        assert "nmpiw" in result
        assert "winkler_score" in result

    def test_compute_all_metrics_with_climatology(self):
        """Test compute_all_metrics with climatology."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 5.1])
        y_clim = np.array([3.0, 3.0, 3.0, 3.0, 3.0])

        result = compute_all_metrics(y_true, y_pred, y_climatology=y_clim)

        assert "skill_score" in result


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_arrays(self):
        """Test with empty arrays."""
        y_true = np.array([])
        y_pred = np.array([])

        for func in [nse, kge, nrmse, mae]:
            result = func(y_true, y_pred)
            assert np.isnan(result) or result == 0.0

    def test_single_value(self):
        """Test with single value."""
        y_true = np.array([1.0])
        y_pred = np.array([1.0])

        result = nse(y_true, y_pred)
        # With single value, NSE is undefined (0/0)
        assert np.isnan(result) or result == 1.0