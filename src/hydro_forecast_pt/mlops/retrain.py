"""
Biweekly retraining pipeline for hydro-forecast-ch.

Orchestrates:
- Data ingestion and feature engineering
- Model training with rolling origin CV
- Evaluation and metric computation
- Champion/challenger model selection
- MLflow tracking and DVC versioning
"""
import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..config import get_config
from ..data.ingestion_pipeline import IngestionPipeline
from ..data.storage import DataLake
from ..features.feature_pipeline import FeaturePipeline
from ..models import create_model
from ..models.base import BaseModel
from ..evaluation import Backtester, RollingOriginSplitter, compute_all_metrics
from .mlflow_utils import MLflowManager, ModelRegistry, RetrainingPipeline

logger = logging.getLogger(__name__)


@dataclass
class RetrainConfig:
    """Retraining configuration."""
    model_types: List[str]
    basins: List[str]
    horizons: List[int]
    initial_train_weeks: int = 104
    step_weeks: int = 13
    max_folds: int = 5
    gap_weeks: int = 2
    champion_metric: str = "skill_score"
    min_improvement: float = 0.01
    output_dir: str = "models/"
    metrics_dir: str = "metrics/"


class RetrainingOrchestrator:
    """
    Orchestrates the full retraining pipeline.
    
    Steps:
    1. Ingest latest data (ENTSO-E, weather, BFE)
    2. Generate features
    3. Train models with rolling origin CV
    4. Evaluate models
    5. Champion/challenger selection
    6. Register and promote models
    7. Version data and models with DVC
    """
    
    def __init__(
        self,
        config: Optional[RetrainConfig] = None,
        mlflow_manager: Optional[MLflowManager] = None,
        model_registry: Optional[ModelRegistry] = None,
    ):
        """
        Initialize orchestrator.
        
        Args:
            config: RetrainConfig instance
            mlflow_manager: MLflowManager instance
            model_registry: ModelRegistry instance
        """
        self.config = config or self._default_config()
        self.mlflow = mlflow_manager or MLflowManager()
        self.registry = model_registry or ModelRegistry(self.mlflow)
        self.data_lake = DataLake()
        
        logger.info(f"RetrainingOrchestrator initialized for models: {self.config.model_types}")
    
    def _default_config(self) -> RetrainConfig:
        """Create default config from global config."""
        cfg = get_config()
        return RetrainConfig(
            model_types=list(cfg.model.keys()),
            basins=cfg.data.basins,
            horizons=cfg.data.horizons,
            initial_train_weeks=cfg.evaluation.initial_train_weeks,
            step_weeks=cfg.evaluation.step_weeks,
            max_folds=cfg.evaluation.max_folds,
            gap_weeks=cfg.evaluation.gap_weeks,
            champion_metric=cfg.mlops.model_registry.champion_metric,
            min_improvement=cfg.mlops.model_registry.min_improvement,
        )
    
    def run_full_retrain(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_ingest: bool = False,
    ) -> Dict[str, Any]:
        """
        Run complete retraining pipeline.
        
        Args:
            start_date: Start date for data (ISO format)
            end_date: End date for data (ISO format)
            force_ingest: Force data re-ingestion
            
        Returns:
            Dictionary with results
        """
        logger.info("=" * 60)
        logger.info("STARTING FULL RETRAINING PIPELINE")
        logger.info("=" * 60)
        
        results = {
            "start_time": datetime.now(),
            "models": {},
            "data_version": None,
            "errors": [],
        }
        
        try:
            # Step 1: Data ingestion
            logger.info("Step 1: Data Ingestion")
            X, y = self._ingest_and_prepare_data(start_date, end_date, force_ingest)
            
            # Step 2: Feature engineering
            logger.info("Step 2: Feature Engineering")
            X_features = self._engineer_features(X)
            
            # Step 3: Prepare targets
            logger.info("Step 3: Prepare Targets")
            y_targets = self._prepare_targets(y, X_features)
            
            # Step 4: Climatology baseline
            logger.info("Step 4: Climatology Baseline")
            X_climatology = self._compute_climatology(X_features, y_targets)
            
            # Step 5: Train and evaluate models
            logger.info("Step 5: Model Training & Evaluation")
            for model_type in self.config.model_types:
                try:
                    model_result = self._train_and_evaluate_model(
                        model_type, X_features, y_targets, X_climatology
                    )
                    results["models"][model_type] = model_result
                except Exception as e:
                    logger.error(f"Model {model_type} failed: {e}")
                    results["errors"].append(f"{model_type}: {str(e)}")
            
            # Step 6: Champion selection
            logger.info("Step 6: Champion Selection")
            self._select_champions(results["models"])
            
            # Step 7: Version data
            logger.info("Step 7: Data Versioning")
            self._version_data()
            
            results["end_time"] = datetime.now()
            results["duration"] = (results["end_time"] - results["start_time"]).total_seconds()
            
            logger.info("=" * 60)
            logger.info("RETRAINING PIPELINE COMPLETED")
            logger.info(f"Duration: {results['duration']:.1f}s")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            results["errors"].append(f"Pipeline: {str(e)}")
            results["end_time"] = datetime.now()
            raise
        
        return results
    
    def _ingest_and_prepare_data(
        self,
        start_date: Optional[str],
        end_date: Optional[str],
        force: bool,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Ingest and prepare training data."""
        # Initialize ingestion pipeline
        pipeline = IngestionPipeline()
        
        # Run ingestion (uses cached data if available and not forced)
        pipeline.run_full_ingestion(
            start_date=start_date,
            end_date=end_date,
            force=force,
        )
        
        # Load processed data from data lake
        X = self.data_lake.load_processed("features")
        y = self.data_lake.load_processed("targets")
        
        if X is None or y is None:
            raise ValueError("No processed data found. Run ingestion first.")
        
        logger.info(f"Loaded data: X={X.shape}, y={y.shape}")
        return X, y
    
    def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate features."""
        feature_pipeline = FeaturePipeline()
        
        # Fit on training data (up to current date)
        X_features = feature_pipeline.fit_transform(X)
        
        logger.info(f"Generated features: {X_features.shape}")
        return X_features
    
    def _prepare_targets(
        self,
        y: pd.DataFrame,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """Prepare target variables aligned with features."""
        # Align indices
        common_idx = X.index.intersection(y.index)
        y_aligned = y.loc[common_idx].copy()
        logger.info(f"Aligned targets: {y_aligned.shape}")
        return y_aligned
    
    def _compute_climatology(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute climatology baseline predictions."""
        # Simple weekly climatology per basin
        climatology = X.copy()
        
        target_cols = [c for c in y.columns if c.startswith("target_cf_")]
        
        for basin in self.config.basins:
            basin_mask = X["basin_id"] == basin
            
            for h in self.config.horizons:
                col = f"target_cf_h{h}"
                if col in y.columns:
                    # Weekly climatology
                    week_of_year = X.loc[basin_mask, "week_of_year"]
                    clim_values = y.loc[basin_mask, col].groupby(week_of_year).mean()
                    
                    # Map to all rows
                    climatology.loc[basin_mask, col] = week_of_year.map(clim_values).fillna(
                        y.loc[basin_mask, col].mean()
                    )
        
        logger.info("Computed climatology baseline")
        return climatology
    
    def _train_and_evaluate_model(
        self,
        model_type: str,
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_climatology: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Train and evaluate a single model type."""
        logger.info(f"Training {model_type}...")
        
        # Create model
        model = create_model(model_type)
        model.set_params(**get_config().model.get(model_type, {}))
        
        # Create splitter
        splitter = RollingOriginSplitter(
            initial_train_size=self.config.initial_train_weeks,
            step_size=self.config.step_weeks,
            max_folds=self.config.max_folds,
            gap=self.config.gap_weeks,
            horizons=self.config.horizons,
        )
        
        # Backtest
        backtester = Backtester(splitter=splitter)
        result = backtester.run_backtest(model, X, y, X_climatology)
        
        # Log to MLflow
        with self.mlflow.start_run(model_type) as run:
            self.mlflow.log_params(model.get_params())
            self.mlflow.log_metrics(result.aggregated_metrics)
            self.mlflow.log_model(model, f"{model_type}_model", model_type)
            run_id = run.info.run_id
        
        # Register as challenger
        challenger = self.registry.register_challenger(
            model, model_type, run_id,
            result.aggregated_metrics,
            model.get_params(),
        )
        
        # Evaluate against champion
        if self.registry.evaluate_challenger(challenger):
            self.registry.promote_challenger(challenger)
            status = "promoted"
        else:
            status = "rejected"
        
        logger.info(f"{model_type}: {status} (NSE={result.aggregated_metrics.get('nse', 0):.3f})")
        
        return {
            "model_type": model_type,
            "run_id": run_id,
            "challenger_version": challenger.version,
            "metrics": result.aggregated_metrics,
            "horizon_metrics": result.horizon_metrics.to_dict("records") if not result.horizon_metrics.empty else [],
            "status": status,
        }
    
    def _select_champions(self, results: Dict[str, Any]) -> None:
        """Select and promote champion models."""
        for model_type, result in results.items():
            if result.get("status") == "promoted":
                logger.info(f"Champion selected: {model_type} v{result['challenger_version']}")
            else:
                logger.info(f"No champion update for {model_type}")
    
    def _version_data(self) -> None:
        """Version data and models with DVC."""
        try:
            from .dvc_utils import DVCManager
            dvc = DVCManager()
            dvc.add_data(["data/", "models/"])
            dvc.push()
            logger.info("Data and models versioned with DVC")
        except Exception as e:
            logger.warning(f"DVC versioning failed: {e}")


def run_ingestion_only(start_date: str = None, end_date: str = None) -> None:
    """Run only data ingestion."""
    pipeline = IngestionPipeline()
    pipeline.run_full_ingestion(start_date=start_date, end_date=end_date)


def run_features_only() -> None:
    """Run only feature engineering."""
    data_lake = DataLake()
    X = data_lake.load_processed("features")
    if X is not None:
        feature_pipeline = FeaturePipeline()
        X_features = feature_pipeline.fit_transform(X)
        data_lake.save_processed(X_features, "features_engineered")
        logger.info("Feature engineering completed")


def run_training_only(model_types: List[str] = None) -> None:
    """Run only model training."""
    config = get_config()
    model_types = model_types or list(config.model.keys())
    
    data_lake = DataLake()
    X = data_lake.load_processed("features_engineered")
    y = data_lake.load_processed("targets")
    
    if X is None or y is None:
        raise ValueError("No engineered features found. Run features step first.")
    
    orchestrator = RetrainingOrchestrator()
    orchestrator.config.model_types = model_types
    
    # Simple training without full pipeline
    for model_type in model_types:
        model = create_model(model_type)
        model.set_params(**config.model.get(model_type, {}))
        model.fit(X, y)
        logger.info(f"Trained {model_type}")


def select_champion(model_type: str = None) -> None:
    """Run champion selection for specific or all models."""
    config = get_config()
    model_types = [model_type] if model_type else list(config.model.keys())
    
    mlflow = MLflowManager()
    registry = ModelRegistry(mlflow)
    
    for mt in model_types:
        model_name = f"hydro-forecast-ch-{mt}"
        try:
            # Get staging model
            challenger_model, challenger_version = registry.get_challenger(mt)
            
            # Evaluate
            if registry.evaluate_challenger(challenger_version):
                registry.promote_challenger(challenger_version)
                logger.info(f"Promoted {mt} v{challenger_version.version} to champion")
            else:
                logger.info(f"Challenger {mt} v{challenger_version.version} not promoted")
        except Exception as e:
            logger.error(f"Champion selection failed for {mt}: {e}")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(description="Hydro-forecast-ch Retraining Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Full retrain
    full_parser = subparsers.add_parser("full", help="Run full retraining pipeline")
    full_parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    full_parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    full_parser.add_argument("--force", action="store_true", help="Force re-ingestion")
    
    # Ingestion only
    subparsers.add_parser("ingest", help="Run data ingestion only")
    
    # Features only
    subparsers.add_parser("features", help="Run feature engineering only")
    
    # Training only
    train_parser = subparsers.add_parser("train", help="Run model training only")
    train_parser.add_argument("--models", nargs="+", help="Model types to train")
    
    # Champion selection
    champion_parser = subparsers.add_parser("select-champion", help="Select champion model")
    champion_parser.add_argument("--model", type=str, help="Specific model type")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    if args.command == "full":
        orchestrator = RetrainingOrchestrator()
        orchestrator.run_full_retrain(
            start_date=args.start_date,
            end_date=args.end_date,
            force_ingest=args.force,
        )
    elif args.command == "ingest":
        run_ingestion_only(args.start_date, args.end_date)
    elif args.command == "features":
        run_features_only()
    elif args.command == "train":
        run_training_only(args.models)
    elif args.command == "select-champion":
        select_champion(args.model)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

# 2026-03-06: MLOps: skill>baseline+0.01 AND NSE>0.5 AND nRMSE<0.3

# 2026-03-29: MLOps: cron trigger, expanding window, champion/challenger

# 2026-04-02: Add biweekly retraining to retrain: cron trigger, expanding window, champion/cha

# 2026-04-03: Add biweekly retraining to retrain: cron trigger, expanding window, champion/cha

# 2026-04-12: Add champion selection to retrain: skill>baseline+0.01 AND NSE>0.5 AND nRMSE<0.3
