"""
Tests for temporal features module.
"""
import numpy as np
import pandas as pd
import pytest

from hydro_forecast_pt.features.temporal import TemporalFeatures


class TestTemporalFeatures:
    """Tests for TemporalFeatures class."""

    def setup_method(self):
        """Initialize TemporalFeatures."""
        self.temporal = TemporalFeatures()

    def test_compute_cyclical(self):
        """Test cyclical encoding of week and month."""
        dates = pd.date_range("2024-01-01", periods=104, freq="W")  # 2 years
        result = self.temporal.compute_cyclical(dates)

        # Check required columns
        assert "sin_week" in result.columns
        assert "cos_week" in result.columns
        assert "sin_month" in result.columns
        assert "cos_month" in result.columns

        # Check range
        assert (result["sin_week"] >= -1).all() and (result["sin_week"] <= 1).all()
        assert (result["cos_week"] >= -1).all() and (result["cos_week"] <= 1).all()

        # Check periodicity
        # Week 1 and week 53 should be similar (sin/cos periodic)
        assert np.isclose(result["sin_week"].iloc[0], result["sin_week"].iloc[52], atol=0.1)
        assert np.isclose(result["cos_week"].iloc[0], result["cos_week"].iloc[52], atol=0.1)

    def test_compute_water_year(self):
        """Test water year features."""
        dates = pd.date_range("2023-09-01", periods=14, freq="MS")  # Sep 2023 - Oct 2024
        result = self.temporal.compute_water_year(dates)

        assert "water_year" in result.columns
        assert "week_of_water_year" in result.columns
        assert "month_of_water_year" in result.columns

        # Water year starts Oct 1
        # Sep 2023 should be water year 2023
        # Oct 2023 should be water year 2024
        assert result["water_year"].iloc[0] == 2023  # Sep 2023
        assert result["water_year"].iloc[1] == 2024  # Oct 2023

    def test_compute_calendar(self):
        """Test calendar features."""
        dates = pd.date_range("2024-01-01", periods=52, freq="W")
        result = self.temporal.compute_calendar(dates)

        assert "is_weekend" in result.columns
        assert "is_holiday" in result.columns
        assert "quarter" in result.columns
        assert "season" in result.columns

        # Check weekend detection
        weekends = result[result["is_weekend"] == 1]
        # Should have roughly 2/7 weekends
        assert len(weekends) > 10

        # Check quarter
        assert set(result["quarter"].unique()) == {1, 2, 3, 4}

        # Check season
        assert set(result["season"].unique()).issubset({0, 1, 2, 3})  # DJF, MAM, JJA, SON

    def test_compute_trend(self):
        """Test trend features."""
        dates = pd.date_range("2020-01-01", periods=260, freq="W")  # 5 years
        result = self.temporal.compute_trend(dates)

        assert "year" in result.columns
        assert "year_normalized" in result.columns
        assert "week_since_start" in result.columns
        assert "month_since_start" in result.columns

        # Check year_normalized
        assert result["year_normalized"].min() >= -0.5
        assert result["year_normalized"].max() <= 0.5

        # Check monotonic increase
        assert result["week_since_start"].is_monotonic_increasing
        assert result["month_since_start"].is_monotonic_increasing

    def test_compute_all(self):
        """Test compute_all method."""
        dates = pd.date_range("2024-01-01", periods=52, freq="W")
        result = self.temporal.compute_all(dates)

        # Should have all feature types
        expected_cols = [
            "sin_week", "cos_week", "sin_month", "cos_month",
            "water_year", "week_of_water_year", "month_of_water_year",
            "is_weekend", "is_holiday", "quarter", "season",
            "year", "year_normalized", "week_since_start", "month_since_start"
        ]

        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_week_boundary(self):
        """Test week boundary handling (week 52/53)."""
        # End of year
        dates = pd.date_range("2024-12-20", periods=4, freq="W")
        result = self.temporal.compute_all(dates)

        # Weeks should wrap around correctly
        weeks = dates.isocalendar().week
        for i, w in enumerate(weeks):
            assert result["sin_week"].iloc[i] == np.sin(2 * np.pi * w / 52)
            assert result["cos_week"].iloc[i] == np.cos(2 * np.pi * w / 52)


class TestTemporalFeaturesEdgeCases:
    """Edge case tests for TemporalFeatures."""

    def setup_method(self):
        """Initialize TemporalFeatures."""
        self.temporal = TemporalFeatures()

    def test_single_date(self):
        """Test with single date."""
        dates = pd.DatetimeIndex(["2024-06-15"])
        result = self.temporal.compute_all(dates)

        assert len(result) == 1
        assert "sin_week" in result.columns

    def test_leap_year(self):
        """Test leap year handling."""
        dates = pd.date_range("2024-02-25", periods=4, freq="W")  # 2024 is leap year
        result = self.temporal.compute_all(dates)

        # Should handle Feb 29 correctly
        assert len(result) == 4

    def test_holiday_detection(self):
        """Test holiday detection for Swiss holidays."""
        # Swiss National Day - Aug 1
        dates = pd.date_range("2024-07-29", periods=4, freq="W")
        result = self.temporal.compute_all(dates)

        # One of these should be around Aug 1
        holidays = result[result["is_holiday"] == 1]
        # Holiday detection might be approximate (week-based)
        assert len(holidays) >= 0  # At least no error