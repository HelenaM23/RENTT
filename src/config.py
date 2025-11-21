from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from omegaconf import DictConfig

from src.utils import Framework, ScalerType, TaskType


@dataclass
class DatasetConfig:
    name: str
    task_type: str
    num_features: int
    num_output: int
    model_filename: str
    scalar_type: str
    feature_names: List[str]
    loss_func: str

@dataclass
class RENTTConfig:
    random_state: int
    framework: str
    activation_function: str
    hidden_layers: List[int]
    dataset: DatasetConfig
    load_nn_model: bool
    train_test_split_ratio: float
    use_complete_tree: bool

@dataclass
class RENTTFIConfig:
    random_state: int
    framework: str
    activation_function: str
    dataset: DatasetConfig
    train_test_split_ratio: float
    use_complete_tree: bool
    num_samples: Optional[int]
    compute_local_fe: bool
    compute_local_fc: bool
    compute_regional_fe: bool
    compute_regional_fc: bool
    compute_global_fe: bool
    compute_global_fc: bool
    compute_global_comparison_sota_nn_dt: bool
    compute_global_comparison_sota_rentt: bool
    compute_local_comparison_sota_nn_dt: bool
    compute_local_comparison_sota_rentt: bool


class ConfigWrapper:
    """Wrapper for DictConfig with automatic Enum conversion"""
    
    # Mapping from string to Enum
    FRAMEWORK_MAP = {
        "pytorch": Framework.PYTORCH,
        "tensorflow": Framework.TENSORFLOW,
    }
    
    TASK_TYPE_MAP = {
        "regression": TaskType.REGRESSION,
        "classification": TaskType.CLASSIFICATION,
    }
    
    SCALER_TYPE_MAP = {
        "none": ScalerType.NONE,
        "minmax": ScalerType.MINMAX,
        "standard": ScalerType.STANDARD,
    }
    
    def __init__(self, cfg: DictConfig, parent_key: str = ""):
        self._cfg = cfg
        self._parent_key = parent_key
    
    def __getattr__(self, name: str):
        """Get attribute with automatic type conversion"""
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        
        try:
            value = self._cfg[name]
            
            # Handle nested DictConfig - wrap it recursively
            if isinstance(value, DictConfig):
                return ConfigWrapper(value, parent_key=name)
            
            # Convert specific fields to Enums
            if name == "framework" and isinstance(value, str):
                framework_key = value.lower()
                if framework_key not in self.FRAMEWORK_MAP:
                    raise ValueError(
                        f"Unknown framework: '{value}'. "
                        f"Valid options: {list(self.FRAMEWORK_MAP.keys())}"
                    )
                return self.FRAMEWORK_MAP[framework_key]
            
            if name == "task_type" and isinstance(value, str):
                task_key = value.lower()
                if task_key not in self.TASK_TYPE_MAP:
                    raise ValueError(
                        f"Unknown task_type: '{value}'. "
                        f"Valid options: {list(self.TASK_TYPE_MAP.keys())}"
                    )
                return self.TASK_TYPE_MAP[task_key]
            
            if name == "scalar_type" and isinstance(value, str):
                scaler_key = value.lower()
                if scaler_key not in self.SCALER_TYPE_MAP:
                    raise ValueError(
                        f"Unknown scalar_type: '{value}'. "
                        f"Valid options: {list(self.SCALER_TYPE_MAP.keys())}"
                    )
                return self.SCALER_TYPE_MAP[scaler_key]
            
            # Return raw value for everything else
            return value
            
        except KeyError:
            raise AttributeError(
                f"Configuration has no attribute '{name}' "
                f"in {self._parent_key or 'root config'}"
            )
    
    def __getitem__(self, key):
        """Allow dictionary-style access"""
        return self._cfg[key]
    
    def __repr__(self):
        return f"ConfigWrapper({self._parent_key or 'root'})"