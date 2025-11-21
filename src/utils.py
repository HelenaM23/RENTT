import re
import sys
from enum import Enum
from pathlib import Path

import numpy as np
from sympy import Abs


def reshape_if_1d(array: np.ndarray) -> np.ndarray:
    """Reshape array to 2D if it's 1D"""
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def validate_arrays(*arrays: np.ndarray) -> None:
    """Guard: Validate arrays are not None or empty"""
    for i, arr in enumerate(arrays):
        if arr is None:
            print(f"Error: Array {i} is None")
            sys.exit(1)
        if len(arr) == 0:
            print(f"Error: Array {i} is empty")
            sys.exit(1)

def create_directories(dataset_name: str) -> dict:
    """Create necessary directories for transformation"""
    model_dir = Path("Models") / dataset_name
    data_dir = Path("Data") / dataset_name
    feature_importance_dir = Path("Feature_Importance") / dataset_name
    
    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    feature_importance_dir.mkdir(parents=True, exist_ok=True)
        
    return {
        "models": model_dir,
        "data": data_dir,
        "feature_importance": feature_importance_dir,
    }

def extract_hidden_layers_from_filename(model_filename: str) -> list[int, int]:
    """Extract hidden layer dimensions from model filename"""
    numbers = re.findall(r"\d+", model_filename)[0:2]
    return [int(numbers[0]), int(numbers[1])]


def calculate_acc(prediction, y_test, num_samples_test):
    # Classification --> Accuracy
    count = 0
    for i in range(0, num_samples_test):
        if prediction[i] == y_test[i]:
            count += 1

    return count / num_samples_test


def calculate_mse(prediction, y_test, num_samples_test):
    # Regression --> MSE-Error
    mse = 0
    for i in range(0, num_samples_test):
        mse += (prediction[i] - y_test[i]) ** 2

    return mse / num_samples_test


def relu(x):
    return (x + Abs(x)) / 2  # Piecewise((0, x < 0), (x, x >= 0))


class TaskType(str, Enum):
    REGRESSION = 0
    CLASSIFICATION = 1


class Framework(str, Enum):
    PYTORCH = 0
    TENSORFLOW = 1

class ScalerType(str, Enum):
    NONE = "none"
    MINMAX = "minmax"
    STANDARD = "standard"
