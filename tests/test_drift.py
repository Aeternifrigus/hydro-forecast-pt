"""
Tests for drift detection module.
"""
import numpy as np
import pandas as pd
import pytest

from hydro_forecast_pt.mlops.drift import FeatureDriftDetector, DriftResult, PerformanceDriftResult


class TestFeatureDriftDetector:
    """Tests for FeatureDriftDetector."""

    def setup_method(self):
        """Initialize detector."""
        self.detector = FeatureDriftDetector(
            psi_threshold=0.2,
            ks_threshold=0.05,
            js_threshold=0.1,
        )

    def test_psi_no_drift(self):
        """Test PSI with identical distributions."""
        np.random.seed(42)
        ref = np.random.normal(0.5, 0.1, 1000)
        cur = np.random.normal(0.5, 0.1, 1000)

        psi, details = self.detector.psi(ref, cur)
        assert psi >= 0
        assert psi < 0.1  # Should be very low for identical distributions
        assert "bins" in details

    def test_psi_with_drift(self):
        """Test PSI with shifted distribution."""
        np.random.seed(42)
        ref = np.random.normal(0.5, 0.1, 1000)
        cur = np.random.normal(0.8, 0.1, 1000)  # Mean shift

        psi, details = self.detector.psi(ref, cur)
        assert psi > 0.2  # Should detect drift

    def test_psi_with_variance_change(self):
        """Test PSI with variance change."""
        np.random.seed(42)
        ref = np.random.normal(0.5, 0.1, 1000)
        cur = np.random.normal(0.5, 0.3, 1000)  # Higher variance

        psi, details = self.detector.psi(ref, cur)
        assert psi > 0.1  # Should detect some drift

    def test_psi_empty_arrays(self):
        """Test PSI with empty arrays."""
        ref = np.array([])
        cur = np.array([])

        psi, details = self.detector.psi(ref, cur)
        assert psi == 0.0
        assert "error" in details

    def test_psi_nan_handling(self):
        """Test PSI handles NaN values."""
        np.random.seed(42)
        ref = np.random.normal(0.5, 0.1, 1000)
        cur = np.random.normal(0.5, 0.1, 1000)
        cur[100:200] = np.nan  # Insert NaN

        psi, details = self.detector.psi(ref, cur)
        assert psi >= 0

    def test_ks_test_identical(self):
        """Test KS test with identical distributions."""
        np.random.seed(42)
        ref = np.random.normal(0.5, 0.1, 1000)
        cur = np.random.normal(0.5, 0.1, 1000)

        stat, p_value, details = self.detector.ks_test(ref, cur)
        assert stat >= 0
        assert p_value > 0.05  # Should not reject null hypothesis

    def test_ks_test_different(self):
        """Test KS test with different distributions."""
        np.random.seed(42)
        ref = np.random.normal(0.5, 0.1, 1000)
        cur = np.random.normal(0.8, 0.1, 1000)

        stat, p_value, details = self.detector.ks_test(ref, cur)
        assert stat > 0
        assert p_value < 0.001  # Should reject null hypothesis

    def test_jensen_shannon(self):
        """Test Jensen-Shannon divergence."""
        np.random.seed(42)
        ref = np.random.normal(0.5, 0.1, 1000)
        cur = np.random.normal(0.5, 0.1, 1000)

        js, details = self.detector.jensen_shannon(ref, cur)
        assert js >= 0
        assert js < 0.1  # Low for similar distributions

    def test_wasserstein(self):
        """Test Wasserstein distance."""
        np.random.seed(42)
        ref = np.random.normal(0.5, 0.1, 1000)
        cur = np.random.normal(0.5, 0.1, 1000)

        w, details = self.detector.wasserstein_distance(ref, cur)
        assert w >= 0
        assert w < 0.05  # Small for similar distributions


class TestDriftResult:
    """Tests for DriftResult dataclass."""

    def test_drift_result_creation(self):
        """Test DriftResult creation."""
        from datetime import datetime

        result = DriftResult(
            feature="test_feature",
            drift_detected=True,
            drift_score=0.25,
            threshold=0.2,
            method="psi",
            details={"bins": 10},
            timestamp=datetime.now(),
        )

        assert result.feature == "test_feature"
        assert result.drift_detected is True
        assert result.drift_score == 0.25
        assert result.method == "psi"


class TestPerformanceDriftResult:
    """Tests for PerformanceDriftResult dataclass."""

    def test_performance_drift_result(self):
        """Test PerformanceDriftResult creation."""
        from datetime import datetime

        result = PerformanceDriftResult(
            metric="nrmse",
            current_value=0.3,
            baseline_value=0.2,
            degradation=0.1,
            threshold=0.05,
            alert=True,
            timestamp=datetime.now(),
        )

        assert result.metric == "nrmse"
        assert result.degradation == 0.1
        assert result.alert is True