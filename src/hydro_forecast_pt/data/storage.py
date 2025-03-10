"""
Data storage utilities for hydro-forecast-ch.

Handles saving/loading data in Parquet format with partitioning,
checksum verification, and DVC integration.
"""
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..config import get_config, PathsConfig

logger = logging.getLogger(__name__)


class DataLake:
    """
    Data lake interface for storing and retrieving processed data.

    Uses Parquet format with date-based partitioning for efficient querying.
    Integrates with DVC for version control.
    """

    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        """
        Initialize DataLake.

        Args:
            base_path: Base path for data storage. If None, uses config.
        """
        config = get_config()
        self.base_path = Path(base_path or config.paths.processed_data)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Raw data path
        self.raw_path = Path(config.paths.raw_data)
        self.raw_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"DataLake initialized at {self.base_path}")

    def _get_partition_path(self, base: Path, partition_cols: Dict[str, Any]) -> Path:
        """Generate Hive-style partition path from partition columns."""
        parts = []
        for key, value in partition_cols.items():
            if isinstance(value, (list, tuple)):
                value = value[0]
            parts.append(f"{key}={value}")
        return base / "/".join(parts)

    def save_parquet(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        partition_cols: Optional[List[str]] = None,
        compression: str = "snappy",
        overwrite: bool = False,
    ) -> Path:
        """
        Save DataFrame as partitioned Parquet dataset.

        Args:
            df: DataFrame to save
            dataset_name: Name of dataset (subdirectory)
            partition_cols: Columns to partition by (e.g., ["year", "week", "basin"])
            compression: Compression codec (snappy, gzip, zstd)
            overwrite: Whether to overwrite existing data

        Returns:
            Path to saved dataset
        """
        if df.empty:
            logger.warning(f"Empty DataFrame for {dataset_name}, skipping save")
            return self.base_path / dataset_name

        dataset_path = self.base_path / dataset_name

        if overwrite and dataset_path.exists():
            import shutil
            shutil.rmtree(dataset_path)

        dataset_path.mkdir(parents=True, exist_ok=True)

        # Convert to PyArrow table
        table = pa.Table.from_pandas(df)

        # Write partitioned dataset
        if partition_cols:
            pq.write_to_dataset(
                table,
                root_path=str(dataset_path),
                partition_cols=partition_cols,
                compression=compression,
                existing_data_behavior="overwrite_or_ignore",
            )
        else:
            # Single file
            file_path = dataset_path / f"{dataset_name}.parquet"
            pq.write_table(table, str(file_path), compression=compression)

        logger.info(f"Saved {len(df)} rows to {dataset_path}")
        return dataset_path

    def load_parquet(
        self,
        dataset_name: str,
        filters: Optional[List[tuple]] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load Parquet dataset with optional filtering.

        Args:
            dataset_name: Name of dataset to load
            filters: List of (column, operator, value) tuples for partition pruning
            columns: Specific columns to load (None = all)

        Returns:
            Loaded DataFrame
        """
        dataset_path = self.base_path / dataset_name

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        # Use PyArrow dataset for efficient partition pruning
        dataset = pq.ParquetDataset(str(dataset_path), filters=filters)
        table = dataset.read(columns=columns)

        df = table.to_pandas()
        logger.info(f"Loaded {len(df)} rows from {dataset_path}")
        return df

    def load_raw_parquet(
        self,
        dataset_name: str,
        filters: Optional[List[tuple]] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Load from raw data directory."""
        dataset_path = self.raw_path / dataset_name

        if not dataset_path.exists():
            raise FileNotFoundError(f"Raw dataset not found: {dataset_path}")

        dataset = pq.ParquetDataset(str(dataset_path), filters=filters)
        table = dataset.read(columns=columns)
        return table.to_pandas()

    def save_raw_parquet(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        partition_cols: Optional[List[str]] = None,
        compression: str = "snappy",
        overwrite: bool = False,
    ) -> Path:
        """Save to raw data directory."""
        if df.empty:
            logger.warning(f"Empty DataFrame for raw {dataset_name}, skipping save")
            return self.raw_path / dataset_name

        dataset_path = self.raw_path / dataset_name

        if overwrite and dataset_path.exists():
            import shutil
            shutil.rmtree(dataset_path)

        dataset_path.mkdir(parents=True, exist_ok=True)

        table = pa.Table.from_pandas(df)

        if partition_cols:
            pq.write_to_dataset(
                table,
                root_path=str(dataset_path),
                partition_cols=partition_cols,
                compression=compression,
                existing_data_behavior="overwrite_or_ignore",
            )
        else:
            file_path = dataset_path / f"{dataset_name}.parquet"
            pq.write_table(table, str(file_path), compression=compression)

        logger.info(f"Saved {len(df)} raw rows to {dataset_path}")
        return dataset_path

    def compute_checksum(self, file_path: Union[str, Path]) -> str:
        """Compute SHA256 checksum of a file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def verify_checksum(self, file_path: Union[str, Path], expected: str) -> bool:
        """Verify file checksum against expected value."""
        actual = self.compute_checksum(file_path)
        return actual == expected

    def list_datasets(self) -> List[str]:
        """List all available processed datasets."""
        return [d.name for d in self.base_path.iterdir() if d.is_dir()]

    def list_raw_datasets(self) -> List[str]:
        """List all available raw datasets."""
        return [d.name for d in self.raw_path.iterdir() if d.is_dir()]

    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get metadata about a dataset."""
        dataset_path = self.base_path / dataset_name
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        # Read schema from first file
        parquet_files = list(dataset_path.rglob("*.parquet"))
        if not parquet_files:
            return {"path": str(dataset_path), "files": 0, "rows": 0}

        schema = pq.read_schema(str(parquet_files[0]))

        # Estimate row count from metadata
        total_rows = 0
        for pf in parquet_files:
            try:
                meta = pq.read_metadata(str(pf))
                total_rows += meta.num_rows
            except Exception:
                pass

        return {
            "path": str(dataset_path),
            "files": len(parquet_files),
            "estimated_rows": total_rows,
            "columns": list(schema.names),
            "dtypes": {field.name: str(field.type) for field in schema},
        }

    def delete_dataset(self, dataset_name: str, confirm: bool = True) -> bool:
        """Delete a dataset (use with caution)."""
        dataset_path = self.base_path / dataset_name
        if not dataset_path.exists():
            logger.warning(f"Dataset not found: {dataset_path}")
            return False

        if confirm:
            response = input(f"Delete {dataset_path}? [y/N]: ")
            if response.lower() != "y":
                logger.info("Deletion cancelled")
                return False

        import shutil
        shutil.rmtree(dataset_path)
        logger.info(f"Deleted dataset: {dataset_path}")
        return True


class ModelStore:
    """Model artifact storage with versioning."""

    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        config = get_config()
        self.base_path = Path(base_path or config.paths.models)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save_model(
        self,
        model: Any,
        model_name: str,
        version: str,
        metadata: Optional[Dict] = None,
    ) -> Path:
        """Save model with version."""
        import joblib

        model_dir = self.base_path / model_name / f"v{version}"
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / "model.pkl"
        joblib.dump(model, model_path)

        # Save metadata
        if metadata:
            import json
            meta_path = model_dir / "metadata.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Saved model {model_name} v{version} to {model_path}")
        return model_path

    def load_model(self, model_name: str, version: str) -> Any:
        """Load model by name and version."""
        import joblib

        model_path = self.base_path / model_name / f"v{version}" / "model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        return joblib.load(model_path)

    def load_latest(self, model_name: str) -> tuple[Any, str]:
        """Load latest version of a model."""
        model_dir = self.base_path / model_name
        if not model_dir.exists():
            raise FileNotFoundError(f"Model not found: {model_name}")

        versions = sorted([d.name for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("v")])
        if not versions:
            raise FileNotFoundError(f"No versions found for {model_name}")

        latest_version = versions[-1][1:]  # Remove 'v' prefix
        return self.load_model(model_name, latest_version), latest_version

    def list_models(self) -> Dict[str, List[str]]:
        """List all models and their versions."""
        models = {}
        for model_dir in self.base_path.iterdir():
            if model_dir.is_dir():
                versions = sorted([
                    d.name[1:] for d in model_dir.iterdir()
                    if d.is_dir() and d.name.startswith("v")
                ])
                models[model_dir.name] = versions
        return models


# Convenience functions
def get_data_lake() -> DataLake:
    """Get default DataLake instance."""
    return DataLake()


def get_model_store() -> ModelStore:
    """Get default ModelStore instance."""
    return ModelStore()