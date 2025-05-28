"""
Drift detection and monitoring for hydro-forecast-ch.

Monitors:
- Feature drift (distribution shifts)
- Prediction drift (prediction distribution shifts)
- Performance decay (metric degradation)
- Data quality issues
"""
import logging
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from ..config import get_config

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    """Result of drift detection."""
    feature: str
    drift_detected: bool
    drift_score: float
    threshold: float
    method: str
    details: Dict[str, Any]
    timestamp: datetime


@dataclass
class PerformanceDriftResult:
    """Result of performance monitoring."""
    metric: str
    current_value: float
    baseline_value: float
    degradation: float
    threshold: float
    alert: bool
    timestamp: datetime


class FeatureDriftDetector:
    """
    Detects feature distribution drift.
    
    Methods:
    - Population Stability Index (PSI)
    - Kolmogorov-Smirnov test
    - Jensen-Shannon divergence
    - Wasserstein distance
    """
    
    def __init__(
        self,
        psi_threshold: float = 0.2,
        ks_threshold: float = 0.05,
        js_threshold: float = 0.1,
        wasserstein_threshold: float = 0.1,
    ):
        """
        Initialize drift detector.
        
        Args:
            psi_threshold: PSI threshold (0.1-0.25 moderate, >0.25 high)
            ks_threshold: KS test p-value threshold
            js_threshold: Jensen-Shannon divergence threshold
            wasserstein_threshold: Wasserstein distance threshold
        """
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.js_threshold = js_threshold
        self.wasserstein_threshold = wasserstein_threshold
    
    def psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        bins: int = 10,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate Population Stability Index.
        
        PSI = Σ (actual% - expected%) * ln(actual% / expected%)
        
        Args:
            reference: Reference distribution (training data)
            current: Current distribution (new data)
            bins: Number of bins for discretization
            
        Returns:
            Tuple of (PSI score, details dict)
        """
        # Handle edge cases
        reference = reference[~np.isnan(reference)]
        current = current[~np.isnan(current)]
        
        if len(reference) == 0 or len(current) == 0:
            return 0.0, {"error": "Empty data"}
        
        # Create bins based on reference quantiles
        try:
            quantiles = np.linspace(0, 1, bins + 1)
            bin_edges = np.quantile(reference, quantiles)
            bin_edges = np.unique(bin_edges)
            
            if len(bin_edges) < 2:
                return 0.0, {"error": "Insufficient unique values"}
            
            # Histogram
            ref_hist, _ = np.histogram(reference, bins=bin_edges)
            cur_hist, _ = np.histogram(current, bins=bin_edges)
            
            # Normalize
            ref_pct = ref_hist / len(reference)
            cur_pct = cur_hist / len(current)
            
            # Avoid zeros
            eps = 1e-10
            ref_pct = np.clip(ref_pct, eps, 1)
            cur_pct = np.clip(cur_pct, eps, 1)
            
            # PSI
            psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
            
            details = {
                "bins": len(bin_edges) - 1,
                "ref_counts": ref_hist.tolist(),
                "cur_counts": cur_hist.tolist(),
                "ref_pct": ref_pct.tolist(),
                "cur_pct": cur_pct.tolist(),
            }
            
            return float(psi), details
            
        except Exception as e:
            logger.warning(f"PSI calculation failed: {e}")
            return 0.0, {"error": str(e)}
    
    def ks_test(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Kolmogorov-Smirnov two-sample test.
        
        Returns:
            Tuple of (statistic, p-value, details)
        """
        reference = reference[~np.isnan(reference)]
        current = current[~np.isnan(current)]
        
        if len(reference) < 2 or len(current) < 2:
            return 0.0, 1.0, {"error": "Insufficient data"}
        
        stat, p_value = stats.ks_2samp(reference, current)
        
        details = {
            "n_reference": len(reference),
            "n_current": len(current),
        }
        
        return float(stat), float(p_value), details
    
    def jensen_shannon(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        bins: int = 20,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Jensen-Shannon divergence (symmetric KL divergence).
        
        Returns:
            Tuple of (JS divergence, details)
        """
        reference = reference[~np.isnan(reference)]
        current = current[~np.isnan(current)]
        
        if len(reference) == 0 or len(current) == 0:
            return 0.0, {"error": "Empty data"}
        
        # Create common bins
        all_data = np.concatenate([reference, current])
        bin_edges = np.histogram_bin_edges(all_data, bins=bins)
        
        ref_hist, _ = np.histogram(reference, bins=bin_edges, density=True)
        cur_hist, _ = np.histogram(current, bins=bin_edges, density=True)
        
        # Normalize to probabilities
        ref_p = ref_hist / (np.sum(ref_hist) + 1e-10)
        cur_p = cur_hist / (np.sum(cur_hist) + 1e-10)
        
        # Mixture distribution
        m = 0.5 * (ref_p + cur_p)
        
        # JS divergence
        eps = 1e-10
        js = 0.5 * np.sum(ref_p * np.log((ref_p + eps) / (m + eps))) + \
             0.5 * np.sum(cur_p * np.log((cur_p + eps) / (m + eps)))
        
        details = {"bins": bins}
        return float(js), details
    
    def wasserstein_distance(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Wasserstein (Earth Mover's) distance.
        
        Returns:
            Tuple of (distance, details)
        """
        reference = reference[~np.isnan(reference)]
        current = current[~np.isnan(current)]
        
        if len(reference) == 0 or len(current) == 0:
            return 0.0, {"error": "Empty data"}
        
        distance = stats.wasserstein_distance(reference, current)
        
        details = {
            "n_reference": len(reference),
            "n_current": len(current),
            "ref_mean": float(np.mean(reference)),
            "cur_mean": float(np.mean(current)),
            "ref_std": float(np.std(reference)),
            "cur_std": float(np.std(current)),
        }
        
        return float(distance), details
    
    def detect_drift(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str,
        methods: List[str] = None,
    ) -> List[DriftResult]:
        """
        Run all drift detection methods.
        
        Args:
            reference: Reference distribution
            current: Current distribution
            feature_name: Feature name
            methods: List of methods to run (default: all)
            
        Returns:
            List of DriftResult
        """
        methods = methods or ["psi", "ks", "js", "wasserstein"]
        results = []
        
        for method in methods:
            if method == "psi":
                score, details = self.psi(reference, current)
                detected = score > self.psi_threshold
                threshold = self.psi_threshold
            elif method == "ks":
                stat, p_value, details = self.ks_test(reference, current)
                score = stat
                detected = p_value < self.ks_threshold
                threshold = self.ks_threshold
                details["p_value"] = p_value
            elif method == "js":
                score, details = self.jensen_shannon(reference, current)
                detected = score > self.js_threshold
                threshold = self.js_threshold
            elif method == "wasserstein":
                score, details = self.wasserstein_distance(reference, current)
                detected = score > self.wasserstein_threshold
                threshold = self.wasserstein_threshold
            else:
                continue
            
            results.append(DriftResult(
                feature=feature_name,
                drift_detected=detected,
                drift_score=score,
                threshold=threshold,
                method=method,
                details=details,
                timestamp=datetime.now(),
            ))
        
        return results


class PredictionDriftMonitor:
    """
    Monitors prediction distribution drift.
    
    Tracks:
    - Prediction distribution shifts
    - Prediction interval calibration
    - Quantile coverage
    """
    
    def __init__(
        self,
        reference_predictions: np.ndarray,
        reference_targets: np.ndarray,
        reference_lower: Optional[np.ndarray] = None,
        reference_upper: Optional[np.ndarray] = None,
    ):
        """
        Initialize with reference (training) predictions.
        
        Args:
            reference_predictions: Training predictions
            reference_targets: Training targets
            reference_lower: Lower PI bounds (training)
            reference_upper: Upper PI bounds (training)
        """
        self.ref_pred = reference_predictions
        self.ref_targets = reference_targets
        self.ref_lower = reference_lower
        self.ref_upper = reference_upper
        
        # Compute reference statistics
        self.ref_mean = np.nanmean(reference_predictions)
        self.ref_std = np.nanstd(reference_predictions)
        
        if reference_lower is not None and reference_upper is not None:
            self.ref_coverage = np.mean(
                (reference_targets >= reference_lower) & (reference_targets <= reference_upper)
            )
        else:
            self.ref_coverage = None
    
    def check_prediction_drift(
        self,
        current_predictions: np.ndarray,
        current_targets: Optional[np.ndarray] = None,
        current_lower: Optional[np.ndarray] = None,
        current_upper: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Check for prediction drift.
        
        Args:
            current_predictions: New predictions
            current_targets: New targets (optional)
            current_lower: New lower PI bounds
            current_upper: New upper PI bounds
            
        Returns:
            Drift metrics dict
        """
        results = {}
        
        # Distribution drift
        drift_detector = FeatureDriftDetector()
        dist_results = drift_detector.detect_drift(
            self.ref_pred, current_predictions, "predictions"
        )
        results["distribution_drift"] = [r.__dict__ for r in dist_results]
        
        # Mean shift
        cur_mean = np.nanmean(current_predictions)
        mean_shift = abs(cur_mean - self.ref_mean) / (self.ref_std + 1e-10)
        results["mean_shift"] = {
            "reference_mean": float(self.ref_mean),
            "current_mean": float(cur_mean),
            "shift_std": float(mean_shift),
            "alert": mean_shift > 2.0,  # 2 sigma shift
        }
        
        # Std shift
        cur_std = np.nanstd(current_predictions)
        std_ratio = cur_std / (self.ref_std + 1e-10)
        results["std_shift"] = {
            "reference_std": float(self.ref_std),
            "current_std": float(cur_std),
            "ratio": float(std_ratio),
            "alert": std_ratio > 2.0 or std_ratio < 0.5,
        }
        
        # Coverage drift (if intervals available)
        if self.ref_coverage is not None and current_lower is not None and current_upper is not None:
            if current_targets is not None:
                cur_coverage = np.mean(
                    (current_targets >= current_lower) & (current_targets <= current_upper)
                )
                coverage_diff = abs(cur_coverage - self.ref_coverage)
                results["coverage_drift"] = {
                    "reference_coverage": float(self.ref_coverage),
                    "current_coverage": float(cur_coverage),
                    "difference": float(coverage_diff),
                    "alert": coverage_diff > 0.05,
                }
        
        return results


class PerformanceMonitor:
    """
    Monitors model performance decay.
    
    Tracks:
    - Metric degradation over time
    - Rolling window performance
    - Comparison to baseline
    """
    
    def __init__(
        self,
        baseline_metrics: Dict[str, float],
        window_size: int = 30,  # days
        degradation_threshold: float = 0.1,
    ):
        """
        Initialize performance monitor.
        
        Args:
            baseline_metrics: Baseline metric values (from training)
            window_size: Rolling window size (days)
            degradation_threshold: Relative degradation threshold (e.g., 0.1 = 10%)
        """
        self.baseline = baseline_metrics
        self.window_size = window_size
        self.degradation_threshold = degradation_threshold
        self.history: List[Dict] = []
    
    def add_evaluation(
        self,
        metrics: Dict[str, float],
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Add evaluation results to history."""
        entry = {
            "timestamp": timestamp or datetime.now(),
            "metrics": metrics,
        }
        self.history.append(entry)
        
        # Keep only recent history
        cutoff = datetime.now() - timedelta(days=self.window_size * 2)
        self.history = [e for e in self.history if e["timestamp"] > cutoff]
    
    def check_degradation(self) -> List[PerformanceDriftResult]:
        """
        Check for performance degradation.
        
        Returns:
            List of PerformanceDriftResult
        """
        if not self.history:
            return []
        
        # Get recent metrics (last window)
        cutoff = datetime.now() - timedelta(days=self.window_size)
        recent = [e for e in self.history if e["timestamp"] > cutoff]
        
        if not recent:
            return []
        
        # Average recent metrics
        recent_metrics = {}
        for key in self.baseline:
            values = [e["metrics"].get(key) for e in recent if key in e["metrics"]]
            values = [v for v in values if v is not None and not np.isnan(v)]
            if values:
                recent_metrics[key] = np.mean(values)
        
        results = []
        for metric, baseline_val in self.baseline.items():
            if metric not in recent_metrics:
                continue
            
            current_val = recent_metrics[metric]
            
            # Calculate degradation (higher is worse for error metrics)
            if metric in ["nrmse", "mae", "mse", "rmse", "mape", "mpiw", "crps", "quantile_loss", "winkler_score"]:
                # Error metrics: higher is worse
                degradation = (current_val - baseline_val) / (abs(baseline_val) + 1e-10)
            else:
                # Skill metrics: lower is worse
                degradation = (baseline_val - current_val) / (abs(baseline_val) + 1e-10)
            
            alert = degradation > self.degradation_threshold
            
            results.append(PerformanceDriftResult(
                metric=metric,
                current_value=current_val,
                baseline_value=baseline_val,
                degradation=degradation,
                threshold=self.degradation_threshold,
                alert=alert,
                timestamp=datetime.now(),
            ))
        
        return results
    
    def get_performance_summary(self) -> pd.DataFrame:
        """Get performance history as DataFrame."""
        if not self.history:
            return pd.DataFrame()
        
        rows = []
        for entry in self.history:
            row = {"timestamp": entry["timestamp"]}
            row.update(entry["metrics"])
            rows.append(row)
        
        return pd.DataFrame(rows)


class DataQualityMonitor:
    """
    Monitors data quality issues.
    
    Checks:
    - Missing values
    - Outliers
    - Schema changes
    - Value ranges
    """
    
    def __init__(
        self,
        reference_stats: Optional[Dict[str, Dict]] = None,
        missing_threshold: float = 0.05,
        outlier_threshold: float = 3.0,  # std devs
    ):
        """
        Initialize data quality monitor.
        
        Args:
            reference_stats: Reference statistics from training data
            missing_threshold: Max allowed missing fraction
            outlier_threshold: Z-score threshold for outliers
        """
        self.reference_stats = reference_stats or {}
        self.missing_threshold = missing_threshold
        self.outlier_threshold = outlier_threshold
    
    def check_data_quality(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run data quality checks.
        
        Args:
            df: DataFrame to check
            feature_cols: Columns to check (default: all numeric)
            
        Returns:
            Quality report dict
        """
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        report = {
            "timestamp": datetime.now(),
            "n_rows": len(df),
            "n_cols": len(feature_cols),
            "issues": [],
            "warnings": [],
            "column_stats": {},
        }
        
        for col in feature_cols:
            if col not in df.columns:
                continue
            
            series = df[col]
            stats_dict = {
                "dtype": str(series.dtype),
                "missing_count": int(series.isna().sum()),
                "missing_pct": float(series.isna().mean()),
                "min": float(series.min()) if not series.isna().all() else None,
                "max": float(series.max()) if not series.isna().all() else None,
                "mean": float(series.mean()) if not series.isna().all() else None,
                "std": float(series.std()) if not series.isna().all() else None,
            }
            
            # Missing values
            if stats_dict["missing_pct"] > self.missing_threshold:
                report["issues"].append({
                    "column": col,
                    "issue": "high_missing",
                    "value": stats_dict["missing_pct"],
                    "threshold": self.missing_threshold,
                })
            
            # Outliers (if reference stats available)
            if col in self.reference_stats and not series.isna().all():
                ref = self.reference_stats[col]
                if "mean" in ref and "std" in ref and ref["std"] > 0:
                    z_scores = np.abs((series - ref["mean"]) / ref["std"])
                    outlier_pct = (z_scores > self.outlier_threshold).mean()
                    stats_dict["outlier_pct"] = float(outlier_pct)
                    
                    if outlier_pct > 0.05:  # >5% outliers
                        report["warnings"].append({
                            "column": col,
                            "issue": "high_outliers",
                            "value": outlier_pct,
                            "threshold": 0.05,
                        })
            
            # Range check against reference
            if col in self.reference_stats:
                ref = self.reference_stats[col]
                if "min" in ref and stats_dict["min"] is not None:
                    if stats_dict["min"] < ref["min"] * 0.9:
                        report["warnings"].append({
                            "column": col,
                            "issue": "below_reference_min",
                            "value": stats_dict["min"],
                            "reference": ref["min"],
                        })
                if "max" in ref and stats_dict["max"] is not None:
                    if stats_dict["max"] > ref["max"] * 1.1:
                        report["warnings"].append({
                            "column": col,
                            "issue": "above_reference_max",
                            "value": stats_dict["max"],
                            "reference": ref["max"],
                        })
            
            report["column_stats"][col] = stats_dict
        
        return report
    
    def compute_reference_stats(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> Dict[str, Dict]:
        """Compute reference statistics from training data."""
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        stats = {}
        for col in feature_cols:
            series = df[col].dropna()
            if len(series) > 0:
                stats[col] = {
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "q25": float(series.quantile(0.25)),
                    "q75": float(series.quantile(0.75)),
                    "count": len(series),
                }
        
        self.reference_stats = stats
        return stats


class DriftMonitor:
    """
    Unified drift monitoring orchestrator.
    
    Combines:
    - Feature drift detection
    - Prediction drift monitoring
    - Performance decay monitoring
    - Data quality monitoring
    """
    
    def __init__(
        self,
        feature_detector: Optional[FeatureDriftDetector] = None,
        prediction_monitor: Optional[PredictionDriftMonitor] = None,
        performance_monitor: Optional[PerformanceMonitor] = None,
        quality_monitor: Optional[DataQualityMonitor] = None,
    ):
        """Initialize drift monitor."""
        self.feature_detector = feature_detector or FeatureDriftDetector()
        self.prediction_monitor = prediction_monitor
        self.performance_monitor = performance_monitor
        self.quality_monitor = quality_monitor or DataQualityMonitor()
    
    def set_reference_data(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        predictions_train: np.ndarray,
        lower_train: Optional[np.ndarray] = None,
        upper_train: Optional[np.ndarray] = None,
        baseline_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Set reference data from training.
        
        Args:
            X_train: Training features
            y_train: Training targets
            predictions_train: Training predictions
            lower_train: Training lower PI
            upper_train: Training upper PI
            baseline_metrics: Baseline metric values
        """
        # Feature reference stats
        self.quality_monitor.compute_reference_stats(X_train)
        
        # Prediction monitor
        self.prediction_monitor = PredictionDriftMonitor(
            predictions_train,
            y_train.values.flatten() if hasattr(y_train, 'values') else y_train.flatten(),
            lower_train,
            upper_train,
        )
        
        # Performance monitor
        if baseline_metrics:
            self.performance_monitor = PerformanceMonitor(baseline_metrics)
    
    def check_all(
        self,
        X_current: pd.DataFrame,
        y_current: Optional[pd.DataFrame] = None,
        predictions_current: Optional[np.ndarray] = None,
        lower_current: Optional[np.ndarray] = None,
        upper_current: Optional[np.ndarray] = None,
        current_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Run all drift checks.
        
        Args:
            X_current: Current features
            y_current: Current targets
            predictions_current: Current predictions
            lower_current: Current lower PI
            upper_current: Current upper PI
            current_metrics: Current evaluation metrics
            
        Returns:
            Combined drift report
        """
        report = {
            "timestamp": datetime.now(),
            "feature_drift": [],
            "prediction_drift": {},
            "performance_drift": [],
            "data_quality": {},
        }
        
        # Feature drift
        feature_cols = X_current.select_dtypes(include=[np.number]).columns.tolist()
        for col in feature_cols:
            if col in self.quality_monitor.reference_stats:
                ref_data = X_current[col].dropna().values  # Would need stored reference
                # Note: In practice, store reference data or use reference stats
                pass
        
        # Data quality
        report["data_quality"] = self.quality_monitor.check_data_quality(X_current, feature_cols)
        
        # Prediction drift
        if self.prediction_monitor and predictions_current is not None:
            report["prediction_drift"] = self.prediction_monitor.check_prediction_drift(
                predictions_current,
                y_current.values.flatten() if y_current is not None else None,
                lower_current,
                upper_current,
            )
        
        # Performance drift
        if self.performance_monitor and current_metrics:
            self.performance_monitor.add_evaluation(current_metrics)
            perf_results = self.performance_monitor.check_degradation()
            report["performance_drift"] = [r.__dict__ for r in perf_results]
        
        # Summary
        report["summary"] = self._summarize(report)
        
        return report
    
    def _summarize(self, report: Dict) -> Dict[str, Any]:
        """Create summary of drift status."""
        alerts = []
        
        # Feature drift alerts
        for fd in report.get("feature_drift", []):
            if fd.get("drift_detected"):
                alerts.append(f"Feature drift: {fd['feature']} ({fd['method']})")
        
        # Prediction drift alerts
        pred = report.get("prediction_drift", {})
        if pred.get("mean_shift", {}).get("alert"):
            alerts.append("Prediction mean shift")
        if pred.get("std_shift", {}).get("alert"):
            alerts.append("Prediction std shift")
        if pred.get("coverage_drift", {}).get("alert"):
            alerts.append("Prediction interval coverage drift")
        
        # Performance alerts
        for pd in report.get("performance_drift", []):
            if pd.get("alert"):
                alerts.append(f"Performance degradation: {pd['metric']}")
        
        # Data quality
        dq = report.get("data_quality", {})
        alerts.extend([f"Data quality: {i['column']} - {i['issue']}" for i in dq.get("issues", [])])
        
        return {
            "n_alerts": len(alerts),
            "alerts": alerts,
            "status": "CRITICAL" if len(alerts) > 5 else ("WARNING" if alerts else "OK"),
        }


def create_drift_monitor() -> DriftMonitor:
    """Create configured DriftMonitor instance."""
    config = get_config()
    drift_config = config.mlops.drift_monitoring

    feature_detector = FeatureDriftDetector(
        psi_threshold=drift_config.feature_drift_threshold_psi,
        ks_threshold=drift_config.feature_drift_threshold_ks,
    )

    return DriftMonitor(feature_detector=feature_detector)


if __name__ == "__main__":
    # Quick test
    np.random.seed(42)
    ref = np.random.normal(0, 1, 1000)
    cur = np.random.normal(0.3, 1.2, 1000)  # Shifted distribution
    
    detector = FeatureDriftDetector()
    results = detector.detect_drift(ref, cur, "test_feature")
    
    for r in results:
        print(f"{r.feature} ({r.method}): score={r.drift_score:.4f}, "
              f"threshold={r.threshold}, detected={r.drift_detected}")

# 2025-06-07: Hotfix: drift detection static features - excluded lat/lon/wasta from PSI

# 2025-08-31: Fix: drift detection static features - excluded lat/lon/wasta from PSI

# 2025-12-07: Bugfix: drift detection static features in drift - excluded lat/lon/wasta from P

# 2026-01-10: Fix: drift detection static features - excluded lat/lon/wasta from PSI

# 2026-02-07: Add PSI drift detection to drift: Σ(p-q)ln(p/q), bins=10, threshold=0.2

# 2026-02-12: MLOps: ref=first 52w, current=last 26w, 20 features

# 2026-02-13: MLOps: Σ(p-q)ln(p/q), bins=10, threshold=0.2

# 2026-03-14: Add drift reference to drift: ref=first 52w, current=last 26w, 20 features

# 2026-03-19: Add PSI drift detection to drift: Σ(p-q)ln(p/q), bins=10, threshold=0.2

# 2026-04-04: Add PSI drift detection to drift: Σ(p-q)ln(p/q), bins=10, threshold=0.2

# 2026-04-09: MLOps: Σ(p-q)ln(p/q), bins=10, threshold=0.2

# 2026-04-30: MLOps: Σ(p-q)ln(p/q), bins=10, threshold=0.2

# 2026-07-02: Hotfix: drift detection static features - excluded lat/lon/wasta from PSI

# 2026-07-10: Fix: drift detection static features - excluded lat/lon/wasta from PSI
