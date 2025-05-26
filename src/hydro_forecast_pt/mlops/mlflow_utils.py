"""
MLOps utilities for hydro-forecast-ch.

Provides:
- MLflow experiment tracking and model registry
- DVC data versioning utilities
- Model registry with champion/challenger
- Biweekly retraining pipeline
"""
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mlflow
import mlflow.sklearn
import mlflow.pytorch
import mlflow.lightgbm
from mlflow.tracking import MlflowClient

import numpy as np
import pandas as pd

from ..config import get_config
from ..models.base import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Model version metadata."""
    name: str
    version: str
    run_id: str
    stage: str  # "staging", "production", "archived"
    metrics: Dict[str, float]
    params: Dict[str, Any]
    created_at: datetime
    description: str = ""


class MLflowManager:
    """
    Manages MLflow tracking and model registry.

    Handles:
    - Experiment creation
    - Run logging (params, metrics, artifacts)
    - Model registration and versioning
    - Stage transitions (staging -> production)
    - Model loading from registry
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_prefix: str = "hydro-forecast-ch",
        artifact_location: Optional[str] = None,
    ):
        """
        Initialize MLflow manager.

        Args:
            tracking_uri: MLflow tracking URI
            experiment_prefix: Prefix for experiment names
            artifact_location: Artifact storage location
        """
        config = get_config()

        self.tracking_uri = tracking_uri or config.mlops.mlflow.tracking_uri
        self.experiment_prefix = experiment_prefix
        self.artifact_location = artifact_location or config.mlops.mlflow.artifact_location

        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

        logger.info(f"MLflowManager initialized: {self.tracking_uri}")

    def get_or_create_experiment(self, model_type: str) -> str:
        """
        Get or create experiment for model type.

        Args:
            model_type: Type of model (e.g., "random_forest", "lightgbm")

        Returns:
            Experiment ID
        """
        exp_name = f"{self.experiment_prefix}_{model_type}"

        experiment = self.client.get_experiment_by_name(exp_name)
        if experiment:
            return experiment.experiment_id
        else:
            exp_id = self.client.create_experiment(
                exp_name,
                artifact_location=self.artifact_location,
            )
            logger.info(f"Created experiment: {exp_name} (ID: {exp_id})")
            return exp_id

    def start_run(
        self,
        model_type: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> mlflow.ActiveRun:
        """
        Start MLflow run.

        Args:
            model_type: Model type for experiment
            run_name: Optional run name
            tags: Additional tags

        Returns:
            Active MLflow run
        """
        exp_id = self.get_or_create_experiment(model_type)

        tags = tags or {}
        tags.update({
            "model_type": model_type,
            "project": "hydro-forecast-ch",
        })

        run = mlflow.start_run(
            experiment_id=exp_id,
            run_name=run_name,
            tags=tags,
        )

        logger.info(f"Started MLflow run: {run.info.run_id}")
        return run

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics."""
        mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log single metric."""
        mlflow.log_metric(key, value, step=step)

    def log_artifact(self, local_path: Union[str, Path], artifact_path: Optional[str] = None) -> None:
        """Log artifact."""
        mlflow.log_artifact(str(local_path), artifact_path)

    def log_artifacts(self, local_dir: Union[str, Path], artifact_path: Optional[str] = None) -> None:
        """Log directory of artifacts."""
        mlflow.log_artifacts(str(local_dir), artifact_path)

    def log_model(
        self,
        model: BaseModel,
        model_name: str,
        model_type: str,
        signature: Optional[Any] = None,
        input_example: Optional[Any] = None,
    ) -> None:
        """
        Log model to MLflow.

        Args:
            model: Trained model
            model_name: Name for registered model
            model_type: Model type (for flavor)
            signature: Model signature
            input_example: Input example
        """
        if model_type == "lightgbm":
            mlflow.lightgbm.log_model(
                model, model_name, signature=signature, input_example=input_example
            )
        elif model_type == "pytorch":
            mlflow.pytorch.log_model(
                model, model_name, signature=signature, input_example=input_example
            )
        else:
            mlflow.sklearn.log_model(
                model, model_name, signature=signature, input_example=input_example
            )

        logger.info(f"Logged model {model_name} as {model_type}")

    def register_model(
        self,
        run_id: str,
        model_name: str,
        model_path: str = "model",
    ) -> ModelVersion:
        """
        Register model from run.

        Args:
            run_id: MLflow run ID
            model_name: Registered model name
            model_path: Path to model artifact in run

        Returns:
            ModelVersion
        """
        model_uri = f"runs:/{run_id}/{model_path}"

        # Register
        mv = mlflow.register_model(model_uri, model_name)

        # Get run details for metadata
        run = self.client.get_run(run_id)

        version = ModelVersion(
            name=model_name,
            version=mv.version,
            run_id=run_id,
            stage="staging",
            metrics=run.data.metrics,
            params=run.data.params,
            created_at=datetime.fromtimestamp(mv.creation_timestamp / 1000),
            description=f"Registered from run {run_id}",
        )

        # Add tags
        self.client.set_model_version_tag(
            model_name, mv.version, "registered_from_run", run_id
        )
        self.client.set_model_version_tag(
            model_name, mv.version, "stage", "staging"
        )

        logger.info(f"Registered model {model_name} version {mv.version}")
        return version

    def transition_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_existing: bool = True,
    ) -> None:
        """
        Transition model version stage.

        Args:
            model_name: Registered model name
            version: Version number
            stage: Target stage ("staging", "production", "archived")
            archive_existing: Archive existing models in target stage
        """
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage.capitalize(),
            archive_existing_versions=archive_existing,
        )

        # Update tag
        self.client.set_model_version_tag(model_name, version, "stage", stage)

        logger.info(f"Transitioned {model_name} v{version} to {stage}")

    def get_latest_model(
        self,
        model_name: str,
        stage: str = "production",
    ) -> Tuple[BaseModel, ModelVersion]:
        """
        Load latest model from registry.

        Args:
            model_name: Registered model name
            stage: Stage to load from

        Returns:
            Tuple of (model, ModelVersion)
        """
        model_uri = f"models:/{model_name}/{stage}"

        # Determine flavor from model name or try all
        try:
            model = mlflow.sklearn.load_model(model_uri)
            flavor = "sklearn"
        except Exception:
            try:
                model = mlflow.lightgbm.load_model(model_uri)
                flavor = "lightgbm"
            except Exception:
                model = mlflow.pytorch.load_model(model_uri)
                flavor = "pytorch"

        # Get version info
        versions = self.client.get_latest_versions(model_name, stages=[stage.capitalize()])
        if not versions:
            raise ValueError(f"No model found in {stage} stage")

        mv = versions[0]
        run = self.client.get_run(mv.run_id)

        version = ModelVersion(
            name=model_name,
            version=mv.version,
            run_id=mv.run_id,
            stage=stage,
            metrics=run.data.metrics,
            params=run.data.params,
            created_at=datetime.fromtimestamp(mv.creation_timestamp / 1000),
        )

        return model, version

    def list_models(self, filter_string: Optional[str] = None) -> List[ModelVersion]:
        """List registered models."""
        models = []
        for rm in self.client.search_registered_models(filter_string=filter_string):
            for mv in rm.latest_versions:
                run = self.client.get_run(mv.run_id)
                models.append(ModelVersion(
                    name=rm.name,
                    version=mv.version,
                    run_id=mv.run_id,
                    stage=mv.current_stage,
                    metrics=run.data.metrics,
                    params=run.data.params,
                    created_at=datetime.fromtimestamp(mv.creation_timestamp / 1000),
                ))
        return models

    def end_run(self, status: str = "FINISHED") -> None:
        """End current MLflow run."""
        mlflow.end_run(status=status)


class DVCManager:
    """
    DVC utilities for data versioning.

    Handles:
    - DVC initialization
    - Adding data to DVC
    - Pushing/pulling from remote
    - Pipeline reproduction
    """

    def __init__(
        self,
        repo_path: Optional[Union[str, Path]] = None,
        remote: Optional[str] = None,
    ):
        """
        Initialize DVC manager.

        Args:
            repo_path: Path to DVC repository
            remote: DVC remote name
        """
        config = get_config()

        self.repo_path = Path(repo_path or ".")
        self.remote = remote or config.mlops.dvc.remote

        logger.info(f"DVCManager initialized: {self.repo_path}")

    def init(self, force: bool = False) -> None:
        """Initialize DVC repository."""
        import subprocess
        cmd = ["dvc", "init"]
        if force:
            cmd.append("--force")
        result = subprocess.run(cmd, cwd=self.repo_path, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(f"DVC init failed: {result.stderr}")
        else:
            logger.info("DVC initialized")

    def add(self, paths: Union[str, List[str]]) -> None:
        """Add files to DVC."""
        import subprocess
        paths = [paths] if isinstance(paths, str) else paths
        cmd = ["dvc", "add"] + paths
        result = subprocess.run(cmd, cwd=self.repo_path, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"DVC add failed: {result.stderr}")
        else:
            logger.info(f"Added to DVC: {paths}")

    def push(self, targets: Optional[List[str]] = None) -> None:
        """Push to DVC remote."""
        import subprocess
        cmd = ["dvc", "push"]
        if targets:
            cmd.extend(targets)
        if self.remote:
            cmd.extend(["-r", self.remote])
        result = subprocess.run(cmd, cwd=self.repo_path, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"DVC push failed: {result.stderr}")
        else:
            logger.info("DVC push completed")

    def pull(self, targets: Optional[List[str]] = None) -> None:
        """Pull from DVC remote."""
        import subprocess
        cmd = ["dvc", "pull"]
        if targets:
            cmd.extend(targets)
        if self.remote:
            cmd.extend(["-r", self.remote])
        result = subprocess.run(cmd, cwd=self.repo_path, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"DVC pull failed: {result.stderr}")
        else:
            logger.info("DVC pull completed")

    def repro(self, pipeline: Optional[str] = None) -> None:
        """Reproduce DVC pipeline."""
        import subprocess
        cmd = ["dvc", "repro"]
        if pipeline:
            cmd.append(pipeline)
        result = subprocess.run(cmd, cwd=self.repo_path, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"DVC repro failed: {result.stderr}")
        else:
            logger.info("DVC repro completed")

    def status(self) -> Dict[str, List[str]]:
        """Get DVC status."""
        import subprocess
        result = subprocess.run(
            ["dvc", "status", "--json"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            import json
            return json.loads(result.stdout)
        return {}

    def checkout(self) -> None:
        """Checkout DVC-tracked files."""
        import subprocess
        result = subprocess.run(
            ["dvc", "checkout"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error(f"DVC checkout failed: {result.stderr}")
        else:
            logger.info("DVC checkout completed")


class ModelRegistry:
    """
    Model registry with champion/challenger management.

    Manages:
    - Multiple model versions per model type
    - Champion (production) model
    - Challenger (staging) models
    - Automatic promotion based on metrics
    """

    def __init__(
        self,
        mlflow_manager: Optional[MLflowManager] = None,
        champion_metric: str = "skill_score",
        min_improvement: float = 0.01,
    ):
        """
        Initialize model registry.

        Args:
            mlflow_manager: MLflowManager instance
            champion_metric: Metric to compare (higher is better)
            min_improvement: Minimum improvement for promotion
        """
        self.mlflow = mlflow_manager or MLflowManager()
        self.champion_metric = champion_metric
        self.min_improvement = min_improvement

    def register_challenger(
        self,
        model: BaseModel,
        model_type: str,
        run_id: str,
        metrics: Dict[str, float],
        params: Dict[str, Any],
    ) -> ModelVersion:
        """
        Register a challenger model.

        Args:
            model: Trained model
            model_type: Model type (e.g., "random_forest")
            run_id: MLflow run ID
            metrics: Evaluation metrics
            params: Model parameters

        Returns:
            ModelVersion
        """
        model_name = f"hydro-forecast-ch-{model_type}"

        version = self.mlflow.register_model(
            run_id=run_id,
            model_name=model_name,
        )

        logger.info(f"Registered challenger: {model_name} v{version.version}")
        return version

    def evaluate_challenger(
        self,
        challenger_version: ModelVersion,
        champion_version: Optional[ModelVersion] = None,
    ) -> bool:
        """
        Evaluate if challenger should become champion.

        Args:
            challenger_version: Challenger model version
            champion_version: Current champion (optional, fetches if not provided)

        Returns:
            True if challenger should be promoted
        """
        model_name = challenger_version.name

        if champion_version is None:
            # Get current champion
            try:
                _, champion_version = self.mlflow.get_latest_model(model_name, "production")
            except Exception:
                logger.info("No current champion, challenger automatically promoted")
                return True

        # Compare metrics
        challenger_score = challenger_version.metrics.get(self.champion_metric, -np.inf)
        champion_score = champion_version.metrics.get(self.champion_metric, -np.inf)

        improvement = challenger_score - champion_score

        logger.info(f"Champion ({champion_version.version}): {self.champion_metric}={champion_score:.4f}")
        logger.info(f"Challenger ({challenger_version.version}): {self.champion_metric}={challenger_score:.4f}")
        logger.info(f"Improvement: {improvement:.4f} (threshold: {self.min_improvement})")

        return improvement >= self.min_improvement

    def promote_challenger(
        self,
        challenger_version: ModelVersion,
    ) -> None:
        """
        Promote challenger to champion.

        Archives current production model.
        """
        model_name = challenger_version.name
        version = challenger_version.version

        # Archive current production
        try:
            prod_versions = self.mlflow.client.get_latest_versions(
                model_name, stages=["Production"]
            )
            for pv in prod_versions:
                self.mlflow.transition_stage(model_name, pv.version, "archived")
        except Exception:
            pass

        # Promote challenger
        self.mlflow.transition_stage(model_name, version, "production")

        logger.info(f"Promoted {model_name} v{version} to production")

    def get_champion(
        self,
        model_type: str,
    ) -> Tuple[BaseModel, ModelVersion]:
        """
        Get current champion model.

        Args:
            model_type: Model type (e.g., "random_forest")

        Returns:
            Tuple of (model, ModelVersion)
        """
        model_name = f"hydro-forecast-ch-{model_type}"
        return self.mlflow.get_latest_model(model_name, "production")

    def get_challenger(
        self,
        model_type: str,
    ) -> Tuple[BaseModel, ModelVersion]:
        """
        Get current challenger model.

        Args:
            model_type: Model type

        Returns:
            Tuple of (model, ModelVersion)
        """
        model_name = f"hydro-forecast-ch-{model_type}"
        return self.mlflow.get_latest_model(model_name, "staging")


class RetrainingPipeline:
    """
    Biweekly retraining pipeline.

    Orchestrates:
    - Data ingestion
    - Feature engineering
    - Model training
    - Evaluation
    - Champion/challenger gate
    - Model promotion
    """

    def __init__(
        self,
        mlflow_manager: Optional[MLflowManager] = None,
        model_registry: Optional[ModelRegistry] = None,
        dvc_manager: Optional[DVCManager] = None,
        config: Optional[Dict] = None,
    ):
        """
        Initialize retraining pipeline.

        Args:
            mlflow_manager: MLflowManager instance
            model_registry: ModelRegistry instance
            dvc_manager: DVCManager instance
            config: Pipeline configuration
        """
        self.config = config or get_config().mlops.retrain
        self.mlflow = mlflow_manager or MLflowManager()
        self.registry = model_registry or ModelRegistry(self.mlflow)
        self.dvc = dvc_manager or DVCManager()

    def run_retrain(
        self,
        model_types: List[str],
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_climatology: Optional[pd.DataFrame] = None,
    ) -> Dict[str, ModelVersion]:
        """
        Run full retraining for specified model types.

        Args:
            model_types: List of model types to retrain
            X: Training features
            y: Training targets
            X_climatology: Climatology for skill score

        Returns:
            Dict of model_type -> ModelVersion
        """
        logger.info(f"Starting retraining for: {model_types}")

        results = {}

        for model_type in model_types:
            logger.info(f"Retraining {model_type}...")

            try:
                # Train model
                model = self._train_model(model_type, X, y)

                # Evaluate
                metrics = self._evaluate_model(model, X, y, X_climatology)

                # Log to MLflow
                with self.mlflow.start_run(model_type) as run:
                    self.mlflow.log_params(model.get_params())
                    self.mlflow.log_metrics(metrics)

                    # Log model
                    self.mlflow.log_model(model, f"{model_type}_model", model_type)

                    run_id = run.info.run_id

                # Register as challenger
                challenger = self.registry.register_challenger(
                    model, model_type, run_id, metrics, model.get_params()
                )

                # Evaluate against champion
                if self.registry.evaluate_challenger(challenger):
                    self.registry.promote_challenger(challenger)
                    logger.info(f"{model_type}: Challenger promoted to champion!")
                else:
                    logger.info(f"{model_type}: Challenger not promoted")

                results[model_type] = challenger

            except Exception as e:
                logger.error(f"Retraining failed for {model_type}: {e}")
                results[model_type] = None

        # Push DVC updates
        self.dvc.push()

        logger.info("Retraining pipeline completed")
        return results

    def _train_model(
        self,
        model_type: str,
        X: pd.DataFrame,
        y: pd.DataFrame,
    ) -> BaseModel:
        """Train a specific model type."""
        from ..models import create_model

        # Get model class and instantiate
        model = create_model(model_type)

        # Configure from config
        model_config = self.config.get(model_type, {})
        model.set_params(**model_config)

        # Train
        model.fit(X, y)

        return model

    def _evaluate_model(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_climatology: Optional[pd.DataFrame],
    ) -> Dict[str, float]:
        """Evaluate model and return metrics."""
        from ..evaluation import compute_all_metrics

        y_pred = model.predict(X)

        # Get target columns
        target_cols = [c for c in y.columns if c.startswith("target_cf_")]
        y_true = y[target_cols].values.flatten()
        y_pred_flat = y_pred.flatten()

        # Climatology
        y_clim = None
        if X_climatology is not None:
            clim_cols = [c for c in X_climatology.columns if c.startswith("target_cf_")]
            if clim_cols:
                y_clim = X_climatology[clim_cols].values.flatten()

        # Compute metrics
        metrics = compute_all_metrics(
            y_true, y_pred_flat, y_climatology=y_clim
        )

        return metrics


def create_mlflow_manager() -> MLflowManager:
    """Create MLflowManager from config."""
    return MLflowManager()


def create_model_registry() -> ModelRegistry:
    """Create ModelRegistry from config."""
    return ModelRegistry()


def create_retraining_pipeline() -> RetrainingPipeline:
    """Create RetrainingPipeline from config."""
    return RetrainingPipeline()