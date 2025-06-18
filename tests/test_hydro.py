"""
Tests for hydro features module.
"""
import numpy as np
import pandas as pd
import pytest

from hydro_forecast_pt.features.hydro import HydroFeatures


class TestHydroFeatures:
    """Tests for HydroFeatures class."""

    def setup_method(self):
        """Initialize HydroFeatures with default params."""
        self.hydro = HydroFeatures(
            awi_alpha=0.85,
            bucket_tau_fast=2,
            bucket_tau_slow=8,
            ddf=3.5,
        )

    def test_awi_computation(self):
        """Test AWI computation with known values."""
        # Simple test: constant precipitation
        precip = pd.Series([10.0] * 10, index=pd.date_range("2024-01-01", periods=10, freq="W"))
        awi = self.hydro.compute_awi(precip)

        # AWI should approach 10.0 * (1 - 0.85) / (1 - 0.85) = 10.0 * 0.15/0.15 = 10?
        # Actually: AWI_t = 0.85 * AWI_{t-1} + 0.15 * P_t
        # With P_t = 10, steady state: AWI = 0.85*AWI + 1.5 => 0.15*AWI = 1.5 => AWI = 10
        assert len(awi) == 10
        # First value should be 0.15 * 10 = 1.5
        assert np.isclose(awi.iloc[0], 1.5, atol=0.01)
        # Later values should approach 10
        assert awi.iloc[-1] > 8.0  # Close to steady state

    def test_awi_zero_precip(self):
        """Test AWI with zero precipitation."""
        precip = pd.Series([0.0] * 5, index=pd.date_range("2024-01-01", periods=5, freq="W"))
        awi = self.hydro.compute_awi(precip)

        # AWI should decay to 0
        assert np.isclose(awi.iloc[0], 0.0, atol=0.01)
        assert np.isclose(awi.iloc[-1], 0.0, atol=0.01)

    def test_awi_single_value(self):
        """Test AWI with single precipitation event."""
        precip = pd.Series([0.0, 0.0, 100.0, 0.0, 0.0],
                          index=pd.date_range("2024-01-01", periods=5, freq="W"))
        awi = self.hydro.compute_awi(precip)

        assert len(awi) == 5
        # AWI should spike at week 3 then decay
        assert awi.iloc[2] > awi.iloc[1]  # Spike
        assert awi.iloc[3] < awi.iloc[2]  # Decay
        assert awi.iloc[4] < awi.iloc[3]  # Further decay

    def test_bucket_model(self):
        """Test two-layer bucket model."""
        precip = pd.Series([10.0] * 20, index=pd.date_range("2024-01-01", periods=20, freq="W"))
        temp = pd.Series([5.0] * 20, index=pd.date_range("2024-01-01", periods=20, freq="W"))

        result = self.hydro.compute_bucket_model(precip, temp)

        assert "fast_storage" in result
        assert "slow_storage" in result
        assert "fast_runoff" in result
        assert "slow_runoff" in result
        assert "total_runoff" in result

        # Check lengths
        for key, val in result.items():
            assert len(val) == 20, f"{key} has wrong length"

        # Total runoff should be positive
        assert (result["total_runoff"] >= 0).all()

    def test_bucket_model_no_precip(self):
        """Test bucket model with zero precipitation."""
        precip = pd.Series([0.0] * 10, index=pd.date_range("2024-01-01", periods=10, freq="W"))
        temp = pd.Series([5.0] * 10, index=pd.date_range("2024-01-01", periods=10, freq="W"))

        result = self.hydro.compute_bucket_model(precip, temp)

        # Storage should decay to 0
        assert np.isclose(result["fast_storage"].iloc[-1], 0.0, atol=0.1)
        assert np.isclose(result["slow_storage"].iloc[-1], 0.0, atol=0.1)
        # Runoff should be near 0
        assert (result["total_runoff"] < 0.01).all()

    def test_snowmelt(self):
        """Test degree-day snowmelt."""
        temp_mean = pd.Series([5.0, 2.0, -1.0, 3.0, 8.0],
                             index=pd.date_range("2024-01-01", periods=5, freq="W"))
        temp_min = pd.Series([3.0, 0.0, -5.0, 1.0, 5.0],
                            index=pd.date_range("2024-01-01", periods=5, freq="W"))
        precip = pd.Series([0.0, 0.0, 10.0, 0.0, 0.0],
                          index=pd.date_range("2024-01-01", periods=5, freq="W"))
        snow_depth = pd.Series([0.0, 0.0, 20.0, 15.0, 5.0],
                              index=pd.date_range("2024-01-01", periods=5, freq="W"))

        melt = self.hydro.compute_snowmelt(temp_mean, temp_min, precip, snow_depth)

        assert len(melt) == 5
        # Melt should occur when temp > 0
        assert melt.iloc[0] > 0  # 5°C -> melt
        assert melt.iloc[1] > 0  # 2°C -> melt
        assert melt.iloc[2] == 0  # -1°C -> no melt
        assert melt.iloc[3] > 0  # 3°C -> melt
        assert melt.iloc[4] > 0  # 8°C -> melt

    def test_snowmelt_no_snow(self):
        """Test snowmelt with no snow depth."""
        temp_mean = pd.Series([5.0, 10.0],
                             index=pd.date_range("2024-01-01", periods=2, freq="W"))
        temp_min = pd.Series([3.0, 8.0],
                            index=pd.date_range("2024-01-01", periods=2, freq="W"))
        precip = pd.Series([0.0, 0.0],
                          index=pd.date_range("2024-01-01", periods=2, freq="W"))
        snow_depth = pd.Series([0.0, 0.0],
                              index=pd.date_range("2024-01-01", periods=2, freq="W"))

        melt = self.hydro.compute_snowmelt(temp_mean, temp_min, precip, snow_depth)

        # Should still compute melt based on temp even without snow depth
        # (degree-day model uses temp only)
        assert melt.iloc[0] > 0
        assert melt.iloc[1] > 0

    def test_degree_days(self):
        """Test CDD/HDD computation."""
        temp = pd.Series([10.0, 20.0, 25.0, 5.0, 0.0],
                        index=pd.date_range("2024-01-01", periods=5, freq="W"))

        cdd, hdd = self.hydro.compute_degree_days(temp, base=15.0)

        assert len(cdd) == 5
        assert len(hdd) == 5

        # CDD: max(0, T - 15)
        assert cdd.iloc[0] == 0  # 10 - 15 = -5 -> 0
        assert cdd.iloc[1] == 5  # 20 - 15 = 5
        assert cdd.iloc[2] == 10  # 25 - 15 = 10
        assert cdd.iloc[3] == 0  # 5 - 15 = -10 -> 0
        assert cdd.iloc[4] == 0  # 0 - 15 = -15 -> 0

        # HDD: max(0, 15 - T)
        assert hdd.iloc[0] == 5  # 15 - 10 = 5
        assert hdd.iloc[1] == 0  # 15 - 20 = -5 -> 0
        assert hdd.iloc[2] == 0  # 15 - 25 = -10 -> 0
        assert hdd.iloc[3] == 10  # 15 - 5 = 10
        assert hdd.iloc[4] == 15  # 15 - 0 = 15


class TestHydroFeaturesCustomParams:
    """Tests with custom parameters."""

    def test_custom_awi_alpha(self):
        """Test AWI with custom alpha."""
        hydro = HydroFeatures(awi_alpha=0.5)
        precip = pd.Series([10.0] * 10, index=pd.date_range("2024-01-01", periods=10, freq="W"))
        awi = hydro.compute_awi(precip)

        # With alpha=0.5, steady state: AWI = 0.5*AWI + 5 => 0.5*AWI = 5 => AWI = 10
        assert awi.iloc[-1] > 9.0

    def test_custom_ddf(self):
        """Test snowmelt with custom DDF."""
        hydro = HydroFeatures(ddf=5.0)
        temp_mean = pd.Series([5.0], index=pd.date_range("2024-01-01", periods=1, freq="W"))
        temp_min = pd.Series([3.0], index=pd.date_range("2024-01-01", periods=1, freq="W"))
        precip = pd.Series([0.0], index=pd.date_range("2024-01-01", periods=1, freq="W"))
        # Pass None for snow_depth to get potential melt (not limited by SWE)
        snow_depth = None

        melt = hydro.compute_snowmelt(temp_mean, temp_min, precip, snow_depth)
        # DDF=5, temp=5 -> melt = 5 * 5 = 25 mm/day * 7 days = 175 mm/week
        assert np.isclose(melt.iloc[0], 175.0, atol=1.0)

    def test_custom_bucket_tau(self):
        """Test bucket model with custom tau."""
        hydro = HydroFeatures(bucket_tau_fast=1, bucket_tau_slow=4)
        precip = pd.Series([10.0] * 20, index=pd.date_range("2024-01-01", periods=20, freq="W"))
        temp = pd.Series([5.0] * 20, index=pd.date_range("2024-01-01", periods=20, freq="W"))

        result = hydro.compute_bucket_model(precip, temp)

        # Faster response with smaller tau
        assert (result["fast_runoff"] > 0).all()
        assert (result["slow_runoff"] > 0).all()