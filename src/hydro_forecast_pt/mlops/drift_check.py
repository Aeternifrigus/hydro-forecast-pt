"""
Drift detection CLI for hydro-forecast-ch.

Runs comprehensive drift checks and outputs JSON report.
Used by GitHub Actions drift monitoring workflow.
"""
import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import get_config
from ..data.storage import DataLake
from ..models.base import BaseModel
from .mlflow_utils import MLflowManager, ModelRegistry
from .drift import (
    FeatureDriftDetector,
    PredictionDriftMonitor,
    PerformanceMonitor,
    DataQualityMonitor,
    DriftMonitor,
    create_drift_monitor,
)

logger = logging.getLogger(__name__)


def load_training_reference(
    data_lake: DataLake,
    model_registry: ModelRegistry,
    model_type: str,
) -> Dict[str, Any]:
    """Load reference data from training."""
    # Load training features
    X_train = data_lake.load_processed("features")
    y_train = data_lake.load_processed("targets")
    
    # Load champion model and its training predictions
    try:
        champion_model, champion_version = model_registry.get_champion(model_type)
        
        # Get training predictions from MLflow run
        mlflow = MLflowManager()
        run = mlflow.client.get_run(champion_version.run_id)
        
        # Load reference predictions from artifacts
        # In practice, these would be stored as MLflow artifacts
        ref_predictions = None
        ref_lower = None
        ref_upper = None
        
        return {
            "X_train": X_train,
            "y_train": y_train,
            "champion_model": champion_model,
            "champion_version": champion_version,
            "ref_predictions": ref_predictions,
            "ref_lower": ref_lower,
            "ref_upper": ref_upper,
            "baseline_metrics": run.data.metrics,
        }
    except Exception as e:
        logger.warning(f"Could not load champion model: {e}")
        return {
            "X_train": X_train,
            "y_train": y_train,
            "champion_model": None,
            "champion_version": None,
            "ref_predictions": None,
            "ref_lower": None,
            "ref_upper": None,
            "baseline_metrics": {},
        }


def load_current_data(data_lake: DataLake) -> Dict[str, Any]:
    """Load current (latest) data."""
    # Load latest features and targets
    X_current = data_lake.load_processed("features_latest")
    y_current = data_lake.load_processed("targets_latest")
    
    # If no "latest" data, use most recent from processed
    if X_current is None:
        X_current = data_lake.load_processed("features")
    if y_current is None:
        y_current = data_lake.load_processed("targets")
    
    return {
        "X_current": X_current,
        "y_current": y_current,
    }


def generate_current_predictions(
    model: BaseModel,
    X_current: pd.DataFrame,
) -> Dict[str, Any]:
    """Generate predictions on current data."""
    if model is None or X_current is None:
        return {
            "predictions": None,
            "lower": None,
            "upper": None,
        }
    
    try:
        predictions = model.predict(X_current)
        
        lower, upper = None, None
        if hasattr(model, "predict_quantiles"):
            quantiles = model.predict_quantiles(X_current, [0.05, 0.5, 0.95])
            lower = quantiles.get(0.05)
            upper = quantiles.get(0.95)
            predictions = quantiles.get(0.5, predictions)
        
        return {
            "predictions": predictions,
            "lower": lower,
            "upper": upper,
        }
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {
            "predictions": None,
            "lower": None,
            "upper": None,
        }


def run_drift_check(
    model_types: List[str],
    output_path: str,
    threshold: str = "warning",
) -> Dict[str, Any]:
    """
    Run comprehensive drift check.
    
    Args:
        model_types: List of model types to check
        output_path: Path to output JSON report
        threshold: Alert threshold ('warning' or 'critical')
        
    Returns:
        Drift report dictionary
    """
    config = get_config()
    data_lake = DataLake()
    mlflow = MLflowManager()
    registry = ModelRegistry(mlflow)
    
    # Initialize drift monitor
    drift_monitor = create_drift_monitor()
    
    # Combined report
    combined_report = {
        "timestamp": datetime.now().isoformat(),
        "models": {},
        "summary": {
            "n_alerts": 0,
            "alerts": [],
            "status": "OK",
        },
    }
    
    all_alerts = []
    
    for model_type in model_types:
        logger.info(f"Checking drift for {model_type}...")
        
        # Load reference data
        ref_data = load_training_reference(data_lake, registry, model_type)
        
        # Load current data
        curr_data = load_current_data(data_lake)
        
        if ref_data["X_train"] is None or curr_data["X_current"] is None:
            logger.warning(f"Insufficient data for {model_type}, skipping")
            continue
        
        # Set reference data in drift monitor
        if ref_data["ref_predictions"] is not None:
            drift_monitor.set_reference_data(
                X_train=ref_data["X_train"],
                y_train=ref_data["y_train"],
                predictions_train=ref_data["ref_predictions"],
                lower_train=ref_data["ref_lower"],
                upper_train=ref_data["ref_upper"],
                baseline_metrics=ref_data["baseline_metrics"],
            )
        
        # Generate current predictions
        pred_data = generate_current_predictions(
            ref_data["champion_model"],
            curr_data["X_current"],
        )
        
        # Run drift check
        report = drift_monitor.check_all(
            X_current=curr_data["X_current"],
            y_current=curr_data["y_current"],
            predictions_current=pred_data["predictions"],
            lower_current=pred_data["lower"],
            upper_current=pred_data["upper"],
            current_metrics=None,  # Would need recent evaluation metrics
        )
        
        # Convert DriftResult objects to dict
        serializable_report = _make_serializable(report)
        combined_report["models"][model_type] = serializable_report
        
        # Collect alerts
        model_alerts = serializable_report.get("summary", {}).get("alerts", [])
        all_alerts.extend([f"{model_type}: {a}" for a in model_alerts])
    
    # Overall summary
    combined_report["summary"] = {
        "n_alerts": len(all_alerts),
        "alerts": all_alerts,
        "status": "CRITICAL" if len(all_alerts) > 5 else ("WARNING" if all_alerts else "OK"),
    }
    
    # Save report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(combined_report, f, indent=2, default=str)
    
    logger.info(f"Drift report saved to {output_path}")
    logger.info(f"Status: {combined_report['summary']['status']} "
               f"({combined_report['summary']['n_alerts']} alerts)")
    
    return combined_report


def _make_serializable(obj: Any) -> Any:
    """Convert object to JSON-serializable format."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, "__dict__"):
        return _make_serializable(obj.__dict__)
    else:
        return str(obj)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Hydro-forecast-ch Drift Check")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model types to check (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="drift_report.json",
        help="Output JSON report path",
    )
    parser.add_argument(
        "--threshold",
        type=str,
        choices=["warning", "critical"],
        default="warning",
        help="Alert threshold",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Get model types from config if not specified
    if args.models is None:
        config = get_config()
        args.models = list(config.model.keys())
    
    try:
        report = run_drift_check(args.models, args.output, args.threshold)
        
        # Exit with error code if critical alerts
        if report["summary"]["status"] == "CRITICAL":
            sys.exit(2)
        elif report["summary"]["status"] == "WARNING":
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Drift check failed: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()
