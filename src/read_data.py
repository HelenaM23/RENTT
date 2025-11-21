"""
Data loading and preprocessing module for various datasets.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import (fetch_california_housing, fetch_covtype,
                              load_diabetes, load_iris, load_wine)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.config import DatasetConfig, RENTTFIConfig
from src.utils import ScalerType, TaskType, reshape_if_1d, validate_arrays

BUILTIN_LOADERS = {
    "CaliforniaHousing": fetch_california_housing,
    "Diabetes_reg": load_diabetes,
    "IRIS": load_iris,
    "WineQuality": load_wine,
    "ForestCoverTypes": fetch_covtype,
}

CSV_EXTRACTORS = {
    "Linear": lambda df: (df.iloc[:, :3].values, df.iloc[:, 3].values),
    "Absolute": lambda df: (df.iloc[:, 0:1].values, df.iloc[:, 1].values),
    "Quadratic": lambda df: (df.iloc[:, 0:1].values, df.iloc[:, 1].values),
    "CarEvaluation": lambda df: (df.iloc[:, :6].values, df.iloc[:, 6].values),
    "Diabetes": lambda df: (df.iloc[:, :8].values, df.iloc[:, 8].values),
}

# =============================================================================

# Data Scaler

# =============================================================================

class DataScaler:
    """Handles data normalization with different scaling methods."""
    
    def __init__(self, scaler_type: ScalerType = ScalerType.MINMAX):
        self.scaler_type = scaler_type
    
    def scale(self, X: np.ndarray) -> np.ndarray:
        """ Scale the input data. """
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        
        if self.scaler_type == ScalerType.NONE:
            return X
        
        scaler = (
            MinMaxScaler() if self.scaler_type == ScalerType.MINMAX
            else StandardScaler() if self.scaler_type == ScalerType.STANDARD else None
        )
        return scaler.fit_transform(X)


# =============================================================================

# Dataset Loader

# =============================================================================

class DatasetLoader:
    """Loads datasets from various sources (built-in sklearn or CSV)."""
    
    def __init__(self, dataset_config: DatasetConfig):
        self.config = dataset_config
    
    def load(
        self,
        csv_path: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ Load dataset from built-in source or CSV file. """
        # Try built-in datasets first
        if self.config.name in BUILTIN_LOADERS:
            return self._load_builtin()
        
        # Try CSV datasets
        if csv_path and self.config.name in CSV_EXTRACTORS:
            return self._load_csv(csv_path)
        
        raise ValueError(
            f"Dataset '{self.config.name}' not found. "
            f"Provide csv_path for CSV datasets or check dataset name."
        )
    
    def _load_builtin(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load built-in sklearn dataset."""
        data = BUILTIN_LOADERS[self.config.name]()
        X = data.data
        
        # Reshape target for regression
        if self.config.task_type == TaskType.REGRESSION:
            y = data.target.reshape(-1, 1)
        else:
            y = data.target
        
        return X, y
    
    def _load_csv(
        self,
        csv_path: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load dataset from CSV file."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        df = pd.read_csv(csv_path, header=None, delimiter=",")
        return CSV_EXTRACTORS[self.config.name](df)


# =============================================================================

# Data Preprocessor

# =============================================================================

class DataPreprocessor:
    """Handles data preprocessing (scaling, type conversion)."""
    
    def __init__(self, scaler_type: str):
        """ Initialize preprocessor. """
        self.scaler = DataScaler(scaler_type)
    
    def preprocess(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_type: TaskType,
        dataset_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ Preprocess features and target. """
        if X is None or y is None:
            raise ValueError(f"No data available for '{dataset_name}'")
        
        # Scale features
        X = self.scaler.scale(X)
        
        if task_type == TaskType.REGRESSION:
            y = self.scaler.scale(y).ravel()
        elif task_type == TaskType.CLASSIFICATION:
            y = y.astype(int)
        else:
            #raise ValueError(f"Unsupported task type: {task_type}")
            pass
        
        return X, y


# =============================================================================

# Main Dataset Manager

# =============================================================================

class DatasetManager:
    """
    Main class for managing dataset loading, preprocessing, and splitting.
    """
    
    def __init__(
        self,
        random_state: int,
        dataset_config: DatasetConfig,
    ):
        """ Initialize DatasetManager. """
        self.loader = DatasetLoader(dataset_config)
        self.preprocessor = DataPreprocessor(dataset_config.scalar_type)
        self.random_state = random_state
        self.config = dataset_config
    
    def prepare_dataset(
        self,
        train_ratio: float,
        csv_path: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ Load, preprocess, and split dataset. """
        # Validate train_ratio
        if not 0.0 < train_ratio < 1.0:
            raise ValueError(
                f"train_ratio must be between 0 and 1, got {train_ratio}"
            )
        
        # Load dataset
        X, y = self.loader.load(csv_path)

        
        # Preprocess
        X, y = self.preprocessor.preprocess(X, y, self.config.task_type, self.config.name)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            train_size=train_ratio,
            random_state=self.random_state
        )
        
        return X_train, X_test, y_train, y_test
    

# =============================================================================
# Helper Functions
# =============================================================================

def combine_train_test_data(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Combine training and test data"""
    X_total = np.concatenate((X_train, X_test), axis=0)
    y_total = np.concatenate((y_train, y_test), axis=0)
    return X_total, y_total


def load_and_prepare_dataset(config: RENTTFIConfig,
                             data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and prepare dataset"""
    print("Loading data...")
    
    manager = DatasetManager(
        random_state=config.random_state,
        dataset_config=config.dataset
    )
    
    # Prepare CSV path (only used if not a built-in dataset)
    csv_path = data_dir / f"{config.dataset.name}.csv"

    
    # Load and prepare dataset
    X_train, X_test, y_train, y_test = manager.prepare_dataset(
        train_ratio=config.train_test_split_ratio,
        csv_path=str(csv_path) if csv_path.exists() else None
    )
    
    # Guard: Validate loaded data
    validate_arrays(X_train, X_test, y_train, y_test)
    
    # Reshape if necessary
    X_train = reshape_if_1d(X_train)
    X_test = reshape_if_1d(X_test)
    
    # Combine data
    X_total, y_total = combine_train_test_data(X_train, X_test, y_train, y_test)
    
    return X_train, X_test, y_train, y_test, X_total, y_total