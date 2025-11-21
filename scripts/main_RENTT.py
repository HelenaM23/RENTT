import datetime
import re
import sys
import time
from pathlib import Path
from typing import Tuple, Union

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.config import ConfigWrapper, DatasetConfig, RENTTFIConfig
from src.DT import CompleteTree, Tree
from src.read_data import DatasetManager, load_and_prepare_dataset
from src.utils import (Framework, TaskType, create_directories,
                       extract_hidden_layers_from_filename)

# ============================================================================

# CONSTANTS (Explaining Variables Pattern)

# ============================================================================

ACCURACY_HEADERS = [
    "nn_RMSE/Accuracy",
    "Pruned_Tree_RMSE/Accuracy",
    "Pruned_tree_total_layers",
    "Pruned_tree_total_nodes",
]

# ============================================================================

# DIRECTORY AND PATH MANAGEMENT

# ============================================================================

def generate_filepath(directory: Path, prefix: str, 
                             hidden_layers: list, model_filename: str, extension: str) -> Path:
    """Generate output file path"""
    layers_str = "-".join(map(str, hidden_layers))
    return directory / f"{prefix}_{layers_str}_{model_filename}.{extension}"




# ============================================================================

# NEURAL NETWORK OPERATIONS

# ============================================================================

def create_neural_network(config: RENTTFIConfig,
                          model_dir: Path, hidden_layers: list):
    """Create neural network based on framework"""
    print("Initializing neural network...")
    
    if config.framework == Framework.TENSORFLOW:
        from src.nn_tensorflow import DNN
    elif config.framework == Framework.PYTORCH:
        from src.nn_pytorch import DNN
    else:
        raise ValueError(f"Unsupported framework: {config.framework}")

    nn = DNN(
        task_type=config.dataset.task_type,
        num_input=config.dataset.num_features,
        hidden_nodes=hidden_layers,
        num_output=config.dataset.num_output,
        directory=str(model_dir),
        timestamp=datetime.datetime.now()
    )
    return nn


def build_and_evaluate_neural_network(nn, config: RENTTFIConfig,
                                 X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> float:
    """Train or load neural network model"""
    print("Training/Loading Neural Network...")

    neural_network_filename = "DNNmodel_" + config.dataset.model_filename + (".h5" if config.framework == Framework.TENSORFLOW else ".pt")
    
    nn.create_model(
        load_existing=config.load_nn_model,
        model_filename=neural_network_filename,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )

    print("Evaluating Neural Network...")    
    accuracy = nn.evaluate(X_test, y_test)
    print(f"Neural Network - Accuracy/RMSE: {accuracy:.4f}, Layers: {X_train.shape[1], nn.hidden_nodes , y_train.shape[1] if y_train.ndim > 1 else 1}, Framework: {config.framework.name}")
    
    return accuracy


# ============================================================================

# DECISION TREE OPERATIONS

# ============================================================================

def create_decision_tree(nn, config: RENTTFIConfig) -> Union[Tree, CompleteTree]:
    """Create decision tree from neural network"""
    print("Initiating Decision Tree...")
    
    layers_info = nn.load_layer_info()
    
    # Select tree type
    TreeClass = CompleteTree if config.use_complete_tree else Tree

    
    start_time = time.perf_counter()
    tree = TreeClass(
        layers_info,
        num_input=config.dataset.num_features,
        activation_function=config.activation_function,
        task_type=config.dataset.task_type.value,
        num_output=config.dataset.num_output,
        framework=config.framework
    )
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f" Decision Tree Initialization Time: {elapsed_time:.2f} seconds")
    
    return tree


def build_and_evaluate_decision_tree(tree: Union[Tree, CompleteTree], X_total: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, config: RENTTFIConfig) -> Tuple[list, float, int, int]:
    """Evaluate decision tree"""
    print("Building Decision Tree...")
    start_time = time.perf_counter()
    tree_model = tree.build_tree(X_total)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f" Decision Tree Building Time: {elapsed_time:.2f} seconds")

    print("Evaluating Decision Tree...")
    accuracy = tree.evaluate(X_test, y_test)
    num_layers = tree_model[-1].col
    num_nodes = len(tree_model)

    if config.use_complete_tree:
        print(f"Complete Tree - Accuracy/RMSE: {accuracy:.4f}, Number of Levels: {num_layers}, Total Number of Nodes: {num_nodes}")
    else:
        print(f"Tree - Accuracy/RMSE: {accuracy:.4f}, Number of Levels: {num_layers}, Total Number of Pruned Nodes: {num_nodes}")
    
    return tree_model, accuracy, num_layers, num_nodes


# ============================================================================

# RESULTS SAVING

# ============================================================================

def save_results(output_file: Path, nn_accuracy: float, tree_accuracy: float,
                num_layers: int, num_nodes: int) -> None:
    """Save transformation results to CSV"""
    results = pd.DataFrame(
        [[nn_accuracy, tree_accuracy, num_layers, num_nodes]],
        columns=ACCURACY_HEADERS
    )
    
    results.to_csv(output_file, index=False, encoding="utf-8")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

@hydra.main(version_base=None, config_path="../config", config_name="RENTT")
def main(cfg: DictConfig):
    try:
        config= ConfigWrapper(cfg)

        print("Configuration:")
        print(f" Dataset: {config.dataset.name}")
        print(f" Load Pretrained Model: {config.load_nn_model}")
        print(f" Train-Test Split Ratio: {config.train_test_split_ratio}")
        print(f" Use Complete Tree: {config.use_complete_tree}")
        
        
        # Extract hidden layers from model filename
        hidden_layers = config.hidden_layers or extract_hidden_layers_from_filename(
            config.dataset.model_filename
        )
        # Setup directories
        directories = create_directories(config.dataset.name)
        
        # Load and prepare data
        X_train, X_test, y_train, y_test, X_total, y_total = load_and_prepare_dataset(
            config, directories["data"]
        )

        nn = create_neural_network(config, directories["models"], hidden_layers)

        nn_accuracy = build_and_evaluate_neural_network(
            nn, config, X_train, X_test, y_train, y_test
        )
        
        # Create and evaluate decision tree
        tree = create_decision_tree(nn, config)
        tree_model, tree_accuracy, num_layers, num_nodes = build_and_evaluate_decision_tree(
            tree, X_total, X_test, y_test, config
        )
        if config.load_nn_model == False:
            if config.use_complete_tree:
                output_file_decision_tree = str(directories["models"]) + "/CompleteTree_" + config.dataset.model_filename + ".pkl"
            else:
                output_file_decision_tree = str(directories["models"]) + "/PrunedTree_" + config.dataset.model_filename + ".pkl"
        else:
            if config.use_complete_tree:
                output_file_decision_tree = str(directories["models"]) + "/CompleteTree_" + re.sub(r'^(DNNmodel_)', '', config.dataset.model_filename).rsplit('.', 1)[0] + ".pkl"
            else:
                output_file_decision_tree = str(directories["models"]) + "/PrunedTree_" + re.sub(r'^(DNNmodel_)', '', config.dataset.model_filename).rsplit('.', 1)[0] + ".pkl"
        
        tree.save_tree(tree_model, output_file_decision_tree)

        # Save results
        output_file_accuracy = generate_filepath(
            directories["models"], "Accuracy_data", hidden_layers, config.dataset.model_filename, "csv"
        )
        save_results(output_file_accuracy, nn_accuracy, tree_accuracy, num_layers, num_nodes)
        
        print(f"\n{'='*70}")
        print("  Transformation completed!")
        print(f"{'='*70}\n")


    except KeyboardInterrupt:
        print("\n\n Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()