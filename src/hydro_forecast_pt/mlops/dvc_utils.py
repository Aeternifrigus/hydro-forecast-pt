"""
DVC utilities for hydro-forecast-ch.

Provides:
- DVC pipeline stage definitions
- Data versioning helpers
- Pipeline reproduction utilities
"""
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from ..config import get_config

logger = logging.getLogger(__name__)


@dataclass
class DVCStage:
    """DVC pipeline stage definition."""
    name: str
    cmd: str
    deps: List[str]
    outs: List[str]
    params: List[str] = None
    metrics: List[str] = None
    plots: List[str] = None
    desc: str = ""


class DVCManager:
    """
    DVC manager for data and pipeline versioning.
    
    Handles:
    - DVC repository initialization
    - Data tracking (raw, processed, features)
    - Model artifact tracking
    - Pipeline definition and reproduction
    - Remote storage configuration
    """
    
    def __init__(
        self,
        repo_path: Union[str, Path] = ".",
        remote_name: str = "s3remote",
        remote_url: Optional[str] = None,
    ):
        """
        Initialize DVC manager.
        
        Args:
            repo_path: Path to DVC repository
            remote_name: Name of DVC remote
            remote_url: Remote storage URL (S3, GCS, Azure, SSH)
        """
        self.repo_path = Path(repo_path).resolve()
        self.remote_name = remote_name
        self.remote_url = remote_url
        self.config = get_config()
        
        logger.info(f"DVCManager initialized: {self.repo_path}")
    
    def init(self, force: bool = False) -> bool:
        """
        Initialize DVC repository.
        
        Args:
            force: Force re-initialization
            
        Returns:
            True if successful
        """
        cmd = ["dvc", "init"]
        if force:
            cmd.append("--force")
        
        result = self._run_cmd(cmd)
        if result.returncode == 0:
            logger.info("DVC repository initialized")
            return True
        else:
            logger.error(f"DVC init failed: {result.stderr}")
            return False
    
    def configure_remote(self, name: str = None, url: str = None) -> bool:
        """
        Configure DVC remote storage.
        
        Args:
            name: Remote name
            url: Remote URL (e.g., s3://bucket/path, gs://bucket/path)
            
        Returns:
            True if successful
        """
        name = name or self.remote_name
        url = url or self.remote_url or self.config.mlops.dvc.remote
        
        if not url:
            logger.warning("No remote URL provided, skipping remote configuration")
            return False
        
        # Add remote
        result = self._run_cmd(["dvc", "remote", "add", "-d", name, url])
        if result.returncode != 0:
            logger.error(f"Failed to add remote: {result.stderr}")
            return False
        
        # Configure credentials if provided
        if url.startswith("s3://"):
            self._configure_s3_remote(name)
        elif url.startswith("gs://"):
            self._configure_gcs_remote(name)
        elif url.startswith("azure://"):
            self._configure_azure_remote(name)
        
        logger.info(f"Configured DVC remote: {name} -> {url}")
        return True
    
    def _configure_s3_remote(self, name: str) -> None:
        """Configure S3 remote credentials."""
        import os
        access_key = os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        
        if access_key and secret_key:
            self._run_cmd(["dvc", "remote", "modify", name, "access_key_id", access_key])
            self._run_cmd(["dvc", "remote", "modify", name, "secret_access_key", secret_key])
    
    def _configure_gcs_remote(self, name: str) -> None:
        """Configure GCS remote credentials."""
        import os
        cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if cred_path:
            self._run_cmd(["dvc", "remote", "modify", name, "credentialpath", cred_path])
    
    def _configure_azure_remote(self, name: str) -> None:
        """Configure Azure remote credentials."""
        import os
        account_name = os.getenv("AZURE_STORAGE_ACCOUNT")
        account_key = os.getenv("AZURE_STORAGE_KEY")
        
        if account_name and account_key:
            self._run_cmd(["dvc", "remote", "modify", name, "accountname", account_name])
            self._run_cmd(["dvc", "remote", "modify", name, "accountkey", account_key])
    
    def add_data(
        self,
        paths: Union[str, List[str]],
        recursive: bool = True,
    ) -> bool:
        """
        Add data files to DVC tracking.
        
        Args:
            paths: File or directory paths to track
            recursive: Track directories recursively
            
        Returns:
            True if successful
        """
        paths = [paths] if isinstance(paths, str) else paths
        
        cmd = ["dvc", "add"]
        if recursive:
            cmd.append("-R")
        cmd.extend(paths)
        
        result = self._run_cmd(cmd)
        if result.returncode == 0:
            logger.info(f"Added to DVC: {paths}")
            return True
        else:
            logger.error(f"DVC add failed: {result.stderr}")
            return False
    
    def remove_data(self, paths: Union[str, List[str]]) -> bool:
        """Remove data from DVC tracking."""
        paths = [paths] if isinstance(paths, str) else paths
        
        cmd = ["dvc", "remove"] + paths
        result = self._run_cmd(cmd)
        
        if result.returncode == 0:
            logger.info(f"Removed from DVC: {paths}")
            return True
        else:
            logger.error(f"DVC remove failed: {result.stderr}")
            return False
    
    def push(self, targets: Optional[List[str]] = None) -> bool:
        """
        Push data to remote storage.
        
        Args:
            targets: Specific targets to push (default: all)
            
        Returns:
            True if successful
        """
        cmd = ["dvc", "push"]
        if targets:
            cmd.extend(targets)
        if self.remote_name:
            cmd.extend(["-r", self.remote_name])
        
        result = self._run_cmd(cmd)
        if result.returncode == 0:
            logger.info("DVC push completed")
            return True
        else:
            logger.error(f"DVC push failed: {result.stderr}")
            return False
    
    def pull(self, targets: Optional[List[str]] = None) -> bool:
        """
        Pull data from remote storage.
        
        Args:
            targets: Specific targets to pull (default: all)
            
        Returns:
            True if successful
        """
        cmd = ["dvc", "pull"]
        if targets:
            cmd.extend(targets)
        if self.remote_name:
            cmd.extend(["-r", self.remote_name])
        
        result = self._run_cmd(cmd)
        if result.returncode == 0:
            logger.info("DVC pull completed")
            return True
        else:
            logger.error(f"DVC pull failed: {result.stderr}")
            return False
    
    def fetch(self, targets: Optional[List[str]] = None) -> bool:
        """
        Fetch data from remote without checkout.
        
        Args:
            targets: Specific targets to fetch
            
        Returns:
            True if successful
        """
        cmd = ["dvc", "fetch"]
        if targets:
            cmd.extend(targets)
        if self.remote_name:
            cmd.extend(["-r", self.remote_name])
        
        result = self._run_cmd(cmd)
        if result.returncode == 0:
            logger.info("DVC fetch completed")
            return True
        else:
            logger.error(f"DVC fetch failed: {result.stderr}")
            return False
    
    def checkout(self, targets: Optional[List[str]] = None) -> bool:
        """
        Checkout DVC-tracked files to working directory.
        
        Args:
            targets: Specific targets to checkout
            
        Returns:
            True if successful
        """
        cmd = ["dvc", "checkout"]
        if targets:
            cmd.extend(targets)
        
        result = self._run_cmd(cmd)
        if result.returncode == 0:
            logger.info("DVC checkout completed")
            return True
        else:
            logger.error(f"DVC checkout failed: {result.stderr}")
            return False
    
    def repro(self, target: Optional[str] = None, force: bool = False) -> bool:
        """
        Reproduce DVC pipeline.
        
        Args:
            target: Specific stage/target to reproduce
            force: Force reproduction even if unchanged
            
        Returns:
            True if successful
        """
        cmd = ["dvc", "repro"]
        if force:
            cmd.append("--force")
        if target:
            cmd.append(target)
        
        result = self._run_cmd(cmd)
        if result.returncode == 0:
            logger.info("DVC repro completed")
            return True
        else:
            logger.error(f"DVC repro failed: {result.stderr}")
            return False
    
    def dag(self) -> str:
        """Get pipeline DAG visualization."""
        result = self._run_cmd(["dvc", "dag"])
        return result.stdout if result.returncode == 0 else result.stderr
    
    def status(self) -> Dict[str, List[str]]:
        """Get DVC status (changed, new, deleted files)."""
        result = self._run_cmd(["dvc", "status", "--json"])
        if result.returncode == 0:
            import json
            return json.loads(result.stdout)
        return {}
    
    def _run_cmd(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Run command in repo directory."""
        return subprocess.run(
            cmd,
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )


def create_pipeline_stages() -> List[DVCStage]:
    """
    Create standard DVC pipeline stages for hydro-forecast-ch.
    
    Returns:
        List of DVCStage definitions
    """
    config = get_config()
    data_config = config.data
    features_config = config.features
    model_config = config.model
    mlops_config = config.mlops
    
    stages = [
        # Stage 1: Data Ingestion
        DVCStage(
            name="ingest_entsoe",
            cmd="python -m hydro_forecast_pt.data.ingestion_pipeline ingest-entsoe",
            deps=[],
            outs=["data/raw/entsoe/"],
            params=["data.entsoe"],
            desc="Fetch ENTSO-E generation and capacity data",
        ),
        DVCStage(
            name="ingest_weather",
            cmd="python -m hydro_forecast_pt.data.ingestion_pipeline ingest-weather",
            deps=[],
            outs=["data/raw/weather/"],
            params=["data.weather"],
            desc="Fetch ERA5-Land/ECMWF weather data",
        ),
        DVCStage(
            name="ingest_bfe",
            cmd="python -m hydro_forecast_pt.data.ingestion_pipeline ingest-bfe",
            deps=[],
            outs=["data/raw/bfe/"],
            params=["data.bfe"],
            desc="Fetch BFE reservoir and production data",
        ),
        
        # Stage 2: Data Processing
        DVCStage(
            name="process_data",
            cmd="python -m hydro_forecast_pt.data.ingestion_pipeline process",
            deps=["data/raw/entsoe/", "data/raw/weather/", "data/raw/bfe/"],
            outs=["data/processed/"],
            params=["data"],
            desc="Process and merge raw data sources",
        ),
        
        # Stage 3: Feature Engineering
        DVCStage(
            name="engineer_features",
            cmd="python -m hydro_forecast_pt.features.feature_pipeline",
            deps=["data/processed/"],
            outs=["data/features/"],
            params=["features"],
            desc="Generate hydro, temporal, and physics features",
        ),
        
        # Stage 4: Model Training
        DVCStage(
            name="train_models",
            cmd="python -m hydro_forecast_pt.mlops.retrain",
            deps=["data/features/"],
            outs=["models/"],
            params=["model", "mlops.retrain"],
            metrics=["metrics/train_metrics.json"],
            plots=["plots/train/"],
            desc="Train all models with rolling origin CV",
        ),
        
        # Stage 5: Evaluation
        DVCStage(
            name="evaluate_models",
            cmd="python -m hydro_forecast_pt.evaluation.backtest evaluate",
            deps=["models/", "data/features/"],
            outs=["metrics/"],
            params=["evaluation"],
            metrics=["metrics/eval_metrics.json"],
            plots=["plots/eval/"],
            desc="Evaluate models with backtesting",
        ),
        
        # Stage 6: Model Selection
        DVCStage(
            name="select_champion",
            cmd="python -m hydro_forecast_pt.mlops.retrain select-champion",
            deps=["metrics/"],
            outs=["models/champion/"],
            params=["mlops.model_registry"],
            metrics=["metrics/champion_metrics.json"],
            desc="Select champion model via challenger gate",
        ),
    ]
    
    return stages


def generate_dvc_yaml(output_path: Union[str, Path] = "dvc.yaml") -> None:
    """
    Generate DVC pipeline YAML file.
    
    Args:
        output_path: Output YAML path
    """
    import yaml
    
    stages = create_pipeline_stages()
    
    dvc_dict = {
        "stages": {}
    }
    
    for stage in stages:
        stage_dict = {
            "cmd": stage.cmd,
            "deps": stage.deps,
            "outs": stage.outs,
        }
        
        if stage.params:
            stage_dict["params"] = stage.params
        if stage.metrics:
            stage_dict["metrics"] = stage.metrics
        if stage.plots:
            stage_dict["plots"] = stage.plots
        if stage.desc:
            stage_dict["desc"] = stage.desc
        
        dvc_dict["stages"][stage.name] = stage_dict
    
    with open(output_path, "w") as f:
        yaml.dump(dvc_dict, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Generated DVC pipeline at {output_path}")


def create_dvcignore(output_path: Union[str, Path] = ".dvcignore") -> None:
    """Create .dvcignore file."""
    ignore_patterns = [
        "# Python",
        "__pycache__/",
        "*.pyc",
        ".pytest_cache/",
        ".coverage",
        "htmlcov/",
        "",
        "# Jupyter",
        ".ipynb_checkpoints/",
        "*.ipynb",
        "",
        "# IDE",
        ".idea/",
        ".vscode/",
        "*.swp",
        "*.swo",
        "",
        "# OS",
        ".DS_Store",
        "Thumbs.db",
        "",
        "# MLflow",
        "mlruns/",
        "",
        "# Logs",
        "*.log",
        "logs/",
        "",
        "# Temporary",
        "tmp/",
        "temp/",
        "*.tmp",
        "",
        "# Model checkpoints (handled by MLflow)",
        "*.ckpt",
        "*.pt",
        "*.pth",
    ]
    
    with open(output_path, "w") as f:
        f.write("\n".join(ignore_patterns))
    
    logger.info(f"Created .dvcignore at {output_path}")


# Convenience functions
def get_dvc_manager() -> DVCManager:
    """Get configured DVCManager instance."""
    return DVCManager()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m hydro_forecast_pt.mlops.dvc_utils <command>")
        print("Commands: generate-yaml, generate-ignore")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == "generate-yaml":
        generate_dvc_yaml()
    elif cmd == "generate-ignore":
        create_dvcignore()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
