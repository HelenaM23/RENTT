import math
import os
import sys
from contextlib import contextmanager
from itertools import combinations

import dalex as dx
import lime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sage
import tensorflow as tf
from krippendorff import alpha
from matplotlib.ticker import ScalarFormatter
from scipy.stats import rankdata
from sklearn.metrics import log_loss, mean_squared_error
from tqdm import tqdm
import torch
import scipy.special

from src.utils import Framework, TaskType


@contextmanager
def suppress_output():
    with open(os.devnull, "w") as fnull:
        old_stdout = sys.stdout
        sys.stdout = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


class RENTTFeatureImportance:
    """Calculate feature importance and contributions from RENTT decision trees."""
    
    def __init__(self, config):
        self.config = config

    # ==================== Global Methods ====================
    
    def calculate_global_feature_effect(self, tree):
        """Calculate global feature effect from RENTT decision tree."""
        
        def calculate_importance_for_output(tree_model, output_idx=None):
            """Aggregate feature importance across all leaf nodes (global)."""
            leaf_nodes = [node for node in reversed(tree_model) if node.is_leaf]
            
            if not leaf_nodes:
                return None, None
            
            total_samples = sum(node.num_samples for node in leaf_nodes)
            
            # Accumulate weighted importance across all leaves
            accumulated = {
                'abs': None,
                'nor': None,
                'abs_intercept': None,
                'nor_intercept': None,
            }
            
            for node in leaf_nodes:
                weights, abs_weights = self._get_weights(node, output_idx)
                
                scaled_abs = node.num_samples * abs_weights[1:]
                scaled_nor = node.num_samples * weights[1:]
                intercept_abs = abs_weights[0] * node.num_samples
                intercept_nor = weights[0] * node.num_samples
                
                if accumulated['abs'] is None:
                    accumulated['abs'] = scaled_abs
                    accumulated['nor'] = scaled_nor
                    accumulated['abs_intercept'] = intercept_abs
                    accumulated['nor_intercept'] = intercept_nor
                else:
                    accumulated['abs'] += scaled_abs
                    accumulated['nor'] += scaled_nor
                    accumulated['abs_intercept'] += intercept_abs
                    accumulated['nor_intercept'] += intercept_nor
            
            # Normalize by total samples and compute signed magnitude
            final_importance = np.sign(accumulated['nor'] / total_samples) * (accumulated['abs'] / total_samples)
            final_intercept = np.sign(accumulated['nor_intercept'] / total_samples) * (accumulated['abs_intercept'] / total_samples)
            
            return final_importance, final_intercept
        
        # Calculate for single or multiple outputs
        if self.config.dataset.num_output == 1:
            return calculate_importance_for_output(tree.tree)
        
        results = [calculate_importance_for_output(tree.tree, i) 
                   for i in range(self.config.dataset.num_output)]
        
        importances, intercepts = zip(*results)
        return list(importances), list(intercepts)
    
    def calculate_global_feature_contribution(self, X_total, tree, y_total, 
                                             local_feature_contribution=None, 
                                             local_feature_contribution_intercept=None):
        """Calculate global feature contribution by aggregating local contributions."""
        
        if local_feature_contribution is None or local_feature_contribution_intercept is None:
            local_feature_contribution, local_feature_contribution_intercept = \
                self.calculate_local_feature_contribution(X_total, tree, None, None)
        
        if self.config.dataset.task_type == TaskType.CLASSIFICATION:
            return self._aggregate_by_class(
                local_feature_contribution, local_feature_contribution_intercept, y_total
            )
        elif self.config.dataset.task_type == TaskType.REGRESSION:
            return self._aggregate_values(
                local_feature_contribution, 
                local_feature_contribution_intercept
            )
        else:
            raise ValueError(f"Unsupported task type: {self.config.dataset.task_type}")

    # ==================== Regional Methods ====================
    
    def calculate_regional_feature_effect(self, tree):
        """Calculate regional (per-leaf) feature effect from RENTT decision tree."""
        
        def calculate_importance_for_output(tree_model, output_idx=None):
            """Calculate feature importance per leaf node (regional)."""
            leaf_nodes = [node for node in reversed(tree_model) if node.is_leaf]
            
            if not leaf_nodes:
                return np.array([]), np.array([])
            
            num_features = self.config.dataset.num_features
            importances = np.zeros((len(leaf_nodes), num_features))
            intercepts = np.zeros(len(leaf_nodes))
            
            for i, node in enumerate(leaf_nodes):
                weights, abs_weights = self._get_weights(node, output_idx)
                importances[i] = np.sign(weights[1:]) * abs_weights[1:]
                intercepts[i] = np.sign(weights[0]) * abs_weights[0]
            
            return importances, intercepts
        
        # Calculate for single or multiple outputs
        if self.config.dataset.num_output == 1:
            return calculate_importance_for_output(tree.tree)
        
        results = [calculate_importance_for_output(tree.tree, i) 
                   for i in range(self.config.dataset.num_output)]
        
        importances, intercepts = zip(*results)
        return list(importances), list(intercepts)
    
    def calculate_regional_feature_contribution(self, samples, tree, 
                                               local_feature_contribution=None, 
                                               local_feature_contribution_intercept=None):
        """Calculate feature contribution aggregated by activation pattern (region)."""
        
        if local_feature_contribution is None or local_feature_contribution_intercept is None:
            local_feature_contribution, local_feature_contribution_intercept = \
                self.calculate_local_feature_contribution(samples, tree, None, None)
        
        # Get activation patterns
        sample_patterns = []
        for sample in samples:
            sample_tree = tree.build_tree(sample.reshape(1, -1))
            sample_patterns.append(sample_tree[-1].activation_pattern)
        sample_patterns = np.array(sample_patterns, dtype=object)
        
        reference_patterns = np.array(
            [list(node.activation_pattern) for node in tree.tree if node.is_leaf],
            dtype=object
        )
        
        # Aggregate contributions per pattern
        regional_contrib = np.zeros((len(reference_patterns), local_feature_contribution.shape[1]))
        regional_intercept = np.zeros(len(reference_patterns))
        
        for i, ref_pattern in enumerate(reference_patterns):
            mask = np.array([np.array_equal(sp, ref_pattern) for sp in sample_patterns])
            
            if not np.any(mask):
                continue
            
            contrib, intercept = self._aggregate_values(
                local_feature_contribution[mask],
                local_feature_contribution_intercept[mask]
            )
            regional_contrib[i] = contrib
            regional_intercept[i] = intercept
        
        return regional_contrib, regional_intercept

    # ==================== Local Methods ====================
    
    def calculate_local_feature_effect(self, samples, tree):
        """Calculate local feature effect for each sample based on leaf node weights."""
        
        num_samples = len(samples)
        num_features = samples.shape[1]
        
        feature_effects = np.zeros((num_samples, num_features))
        intercepts = np.zeros(num_samples)
        
        for i, sample in enumerate(samples):
            leaf_nodes = [node for node in tree.build_tree(sample.reshape(1, -1)) 
                          if node.is_leaf]
            
            if not leaf_nodes:
                raise ValueError("No leaf node found for sample")
            
            leaf_node = leaf_nodes[-1]
            
            if self.config.dataset.task_type == TaskType.REGRESSION:
                weights = leaf_node.effective_weights
                feature_effects[i] = weights[1:]
                intercepts[i] = weights[0]
            
            elif self.config.dataset.task_type == TaskType.CLASSIFICATION:
                weights = leaf_node.effective_weights
                feature_weights = weights[:, 1:]
                predicted_class = np.argmax(np.dot(feature_weights, sample))
                feature_effects[i] = feature_weights[predicted_class]
                intercepts[i] = weights[predicted_class, 0]
            
            else:
                raise ValueError(f"Unsupported task type: {self.config.dataset.task_type}")
        
        return feature_effects, intercepts

    def calculate_local_feature_contribution(self, samples, tree, 
                                            feature_effect=None, intercept=None):
        """Calculate local feature contribution: effect × sample value."""
        
        if feature_effect is None or intercept is None:
            feature_effect, intercept = self.calculate_local_feature_effect(samples, tree)
        
        if feature_effect.shape != samples.shape:
            raise ValueError(
                f"Feature effect shape {feature_effect.shape} does not match samples shape {samples.shape}"
            )
        
        return feature_effect * samples, intercept

    # ==================== Shared Utilities ====================
    
    def _get_weights(self, node, output_idx=None):
        """Extract weights from node."""
        weights = (node.effective_weights[output_idx] 
                   if output_idx is not None 
                   else node.effective_weights)
        return weights, np.abs(weights)
    
    def _aggregate_values(self, values, intercepts):
        """Aggregate values and intercepts using signed magnitude."""
        contrib = self._aggregate_signed_magnitude(values)
        intercept = self._aggregate_signed_magnitude(intercepts[:, np.newaxis])[0]
        return contrib, intercept
    
    def _aggregate_by_class(self, contributions, intercepts, y_total):
        """Aggregate contributions per class."""
        num_outputs = self.config.dataset.num_output
        num_features = contributions.shape[1]
        
        global_contrib = np.zeros((num_outputs, num_features))
        global_intercept = np.zeros(num_outputs)
        
        for class_idx in range(num_outputs):
            mask = y_total == class_idx
            if not np.any(mask):
                continue
            
            global_contrib[class_idx], global_intercept[class_idx] = self._aggregate_values(
                contributions[mask], intercepts[mask]
            )
        
        return list(global_contrib), list(global_intercept)
    
    def _aggregate_signed_magnitude(self, values):
        """Aggregate values as: sign(mean) × mean(abs)."""
        mean_val = np.mean(values, axis=0)
        return np.sign(mean_val) * np.mean(np.abs(values), axis=0)
        

class DalexFeatureImportance:
    def __init__(self, X_train, y_train, config):
        self.X_train = X_train
        self.y_train = y_train
        self.config = config
        self.num_features = len(config.dataset.feature_names)

    @staticmethod
    def extract_value_lime(var_str):
        """Extract numeric value from LIME variable string."""
        if ">" in var_str:
            return float(var_str.split(">")[0])
        elif "<=" in var_str:
            parts = var_str.split("<=")
            return float(parts[0].split("<")[1] if "<" in parts[0] else parts[0])
        elif "<" in var_str:
            return float(var_str.split("<")[1].split("<=")[0])
        raise ValueError(f"Unexpected format: {var_str}")

    def _extract_explanations(self, explainer, X_test, is_classification, lime_mode):
        """Common explanation extraction for NN and DT."""
        lime_vals = np.zeros((len(X_test), self.num_features))
        shap_vals = np.zeros((len(X_test), self.num_features))
        bd_vals = np.zeros((len(X_test), self.num_features))
        lime_intercept = np.zeros(len(X_test))
        shap_intercept = np.zeros(len(X_test))
        bd_intercept = np.zeros(len(X_test))

        for i, X_test_sample in enumerate(tqdm(X_test)):
            with suppress_output():
                lime_exp = explainer.predict_surrogate(
                    X_test_sample,
                    type="lime",
                    mode=lime_mode,
                    num_features=self.config.dataset.num_features
                )
                shap_exp = explainer.predict_parts(X_test_sample, type="shap", B=5)
                bd_exp = explainer.predict_parts(X_test_sample, type="break_down")

            # SHAP
            df_shap = shap_exp.result[shap_exp.result["B"] == 0]
            data = df_shap[["variable_name", "contribution"]].copy().sort_values("variable_name")
            shap_intercept[i] = shap_exp.intercept
            shap_vals[i] = np.array(data["contribution"])

            # BREAK DOWN
            result_bd = bd_exp.result[["variable_name", "variable", "contribution"]]
            bd_intercept[i] = result_bd[result_bd["variable"] == "intercept"]["contribution"].values[0]
            data_filtered = result_bd[
                (result_bd["variable_name"] != "intercept") & 
                (result_bd["variable_name"] != "")
            ]
            data_sorted = data_filtered.astype({"variable_name": int}).sort_values("variable_name")
            bd_vals[i] = np.array(data_sorted["contribution"])

            # LIME
            lime_intercept[i] = self._extract_lime_intercept(lime_exp, is_classification)
            data = lime_exp.result.copy()
            data["sort_value"] = data["variable"].apply(self.extract_value_lime)
            data_sorted = data.sort_values("sort_value")
            lime_vals[i] = np.array(data_sorted["effect"])

        return lime_vals, shap_vals, bd_vals, lime_intercept, shap_intercept, bd_intercept

    def _extract_lime_intercept(self, lime_exp, is_classification):
        """Extract LIME intercept handling different frameworks."""
        try:
            if is_classification:
                if hasattr(lime_exp.intercept, 'values'):
                    return next(iter(lime_exp.intercept.values()))
                return lime_exp.intercept[0] if isinstance(lime_exp.intercept, (list, np.ndarray)) else lime_exp.intercept
            else:
                return lime_exp.intercept[0] if isinstance(lime_exp.intercept, (list, np.ndarray)) else lime_exp.intercept
        except (AttributeError, TypeError, IndexError):
            return lime_exp.as_list()[0][1] if hasattr(lime_exp, "as_list") else 0

    def calculate_local_feature_importance_nn(self, dnn, X_test):     
        X_subset, y_subset = get_subset(self.X_train, self.y_train, self.config.random_state)
        
        # Konvertiere zu NumPy
        if hasattr(X_test, 'cpu'):
            X_test = X_test.cpu().detach().numpy()
        if hasattr(y_subset, 'cpu'):
            y_subset = y_subset.cpu().detach().numpy()
        
        is_classification = self.config.dataset.task_type == TaskType.CLASSIFICATION
        lime_mode = "classification" if is_classification else "regression"
        label = "Classification" if is_classification else "Regression"
        

        if self.config.framework == Framework.PYTORCH:
            def predict_function(model, X):
                if not isinstance(X, np.ndarray):
                    X = np.array(X)
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                if X.dtype != np.float32:
                    X = X.astype(np.float32)
                
                X_tensor = torch.from_numpy(X).to(next(dnn.model.parameters()).device)
                
                with torch.no_grad():
                    preds = dnn.model(X_tensor).cpu().numpy()

                if is_classification:
                    if preds.shape[1] > 1:
                        preds_prob = scipy.special.softmax(preds, axis=1)
                        result = preds_prob[:, 1] if preds.shape[1] == 2 else preds_prob.max(axis=1)
                    else:
                        result = scipy.special.expit(preds.flatten())
                    return result
                
                if preds.ndim > 1 and preds.shape[1] == 1:
                    result = preds.flatten()
                else:
                    result = preds
                return result
            
            with suppress_output():
                explainer = dx.Explainer(
                    model=dnn,
                    data=X_subset,
                    y=y_subset,
                    predict_function=predict_function,
                    label=label
                )
        

        elif self.config.framework == Framework.TENSORFLOW:
            with suppress_output():
                explainer = dx.Explainer(dnn, X_subset, y_subset, label=label)
        
        print('Calculating Dalex on NN')

        return self._extract_explanations(explainer, X_test, is_classification, lime_mode)
    
    def calculate_local_feature_importance_dt(self, tree, X_test):
        """Calculate local feature importance for decision trees."""
        X_subset, y_subset = get_subset(self.X_train, self.y_train, self.config.random_state)
        is_classification = self.config.dataset.task_type == TaskType.CLASSIFICATION
        lime_mode = "classification" if is_classification else "regression"
        label = "Classification" if is_classification else "Regression"

        class TreeWrapper:
            def __init__(self, tree_model, task_type):
                self.tree_model = tree_model
                self.task_type = task_type

            def predict(self, X):
                if not isinstance(X, np.ndarray):
                    X = np.array(X)
                if X.ndim == 1:
                    X = X.reshape(1, -1)

                predictions = [
                    self.tree_model._predict_sample(sample, self.tree_model.build_tree(sample))
                    for sample in X
                ]
                predictions = np.array(predictions)

                if self.task_type == TaskType.REGRESSION:
                    return predictions.flatten() if predictions.ndim > 1 else predictions
                return np.argmax(predictions, axis=1) if predictions.ndim > 1 else predictions

        tree_wrapper = TreeWrapper(tree, self.config.dataset.task_type)

        with suppress_output():
            explainer = dx.Explainer(tree_wrapper, X_subset, y_subset, label=label)

        print('Calculating DALEX on Decision Tree')
        return self._extract_explanations(explainer, X_test, is_classification, lime_mode)


class SageFeatureImportance:
    def __init__(self, X_train, y_train, config):
        self.X_train = X_train
        self.y_train = y_train
        self.config = config

    def _calculate_intercept(self, y_subset, sensitivity_intercept, is_classification):
        """Calculate intercept based on task type."""
        if is_classification:
            unique_classes = np.unique(y_subset)
            class_counts = np.bincount(y_subset.astype(int))[unique_classes]
            class_probs = class_counts / class_counts.sum()
            baseline_preds = np.tile(class_probs, (len(y_subset), 1))
            return log_loss(y_subset, baseline_preds, labels=unique_classes)
        else:
            baseline_preds = np.full_like(y_subset, fill_value=sensitivity_intercept, dtype=np.float64)
            return mean_squared_error(y_subset, baseline_preds)

    @staticmethod
    def _validate_probabilities(probs):
        """Validate and normalize probabilities."""
        # Replace any NaN or inf values
        probs = np.nan_to_num(probs, nan=1e-7, posinf=1-1e-7, neginf=1e-7)
        
        # Clip to valid range
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        
        # Normalize rows to sum to 1
        if probs.ndim == 2:
            row_sums = probs.sum(axis=1, keepdims=True)
            probs = probs / np.where(row_sums > 0, row_sums, 1)
        
        return probs.astype(np.float64)

    def calculate_global_feature_importance_nn(self, dnn, X_test, y_test):
        """Calculate global feature importance using SAGE for neural networks."""
        X_subset, y_subset = get_subset(self.X_train, self.y_train, self.config.random_state)
        is_classification = self.config.dataset.task_type == TaskType.CLASSIFICATION

        def predict_fn(X):
            X = np.array(X, dtype=np.float32)
            if len(X.shape) == 1:
                X = X.reshape(1, -1)

            if self.config.framework == Framework.TENSORFLOW:
                predictions = dnn.model.predict(X, verbose=0)
            elif self.config.framework == Framework.PYTORCH:
                dnn.model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X) if not isinstance(X, torch.Tensor) else X
                    device = next(dnn.model.parameters()).device
                    predictions = dnn.model(X_tensor.to(device)).cpu().numpy()
            else:
                raise ValueError(f"Unsupported framework: {self.config.framework}")

            if is_classification:
                predictions = self._validate_probabilities(predictions)
            
            return predictions.astype(np.float64)

        imputer = sage.MarginalImputer(predict_fn, X_subset)
        estimator = sage.PermutationEstimator(imputer, self.config.dataset.loss_func)
        sensitivity = estimator(X_test)
        sage_values = estimator(X_test, y_test)
        baseline_pred = np.mean(predict_fn(X_subset))
        intercept = self._calculate_intercept(y_subset, baseline_pred, is_classification)

        return (sensitivity.values, sensitivity.std, baseline_pred,
                sage_values.values, sage_values.std, intercept)

    def calculate_global_feature_importance_dt(self, tree, X_test, y_test):
        """Calculate global feature importance using SAGE for decision trees."""
        X_subset, y_subset = get_subset(self.X_train, self.y_train, self.config.random_state)
        is_classification = self.config.dataset.task_type == TaskType.CLASSIFICATION

        def predict_fn(X):
            if isinstance(X, pd.DataFrame):
                X = X.values
            if not isinstance(X, np.ndarray):
                X = np.array(X, dtype=np.float32)
            if X.ndim == 1:
                X = X.reshape(1, -1)

            predictions = np.array([tree._predict_sample(sample, tree.build_tree(sample)) for sample in X])
            
            if is_classification:
                # Ensure 1D array
                predictions = predictions.squeeze()
                
                # Clip raw predictions to [0, 1] range (they should be probabilities)
                predictions = np.clip(predictions, 0, 1)
                
                # Build probability matrix [P(class 0), P(class 1)]
                probs_pos = predictions
                probs_neg = 1 - probs_pos
                probs = np.column_stack([probs_neg, probs_pos])
                
                # Validate and normalize
                probs = SageFeatureImportance._validate_probabilities(probs)
                
                return probs
            else:
                # For regression, return 1D array
                return predictions.astype(np.float64)

        imputer = sage.MarginalImputer(predict_fn, X_subset)
        estimator = sage.PermutationEstimator(imputer, self.config.dataset.loss_func)
        sensitivity = estimator(X_test)
        sage_values = estimator(X_test, y_test)
        baseline_pred = np.mean(predict_fn(X_subset))
        intercept = self._calculate_intercept(y_subset, baseline_pred, is_classification)

        return (sensitivity.values, sensitivity.std, baseline_pred,
                sage_values.values, sage_values.std, intercept)
    

def get_subset(X_train, y_train, random_seed, max_samples=1000):
    """Create subset of training data"""
    np.random.seed(random_seed)
    if len(X_train) > max_samples:
        indices = np.random.choice(len(X_train), max_samples, replace=False)
        return X_train[indices], y_train[indices]
    return X_train, y_train


def round_to_second_significant(x):
    if x == 0:
        return 0.0
    exponent = int(np.floor(np.log10(abs(x))))
    shift = exponent - 1
    factor = 10**shift
    result = round(x / factor) * factor
    return float(result)


def sci_notation(num, decimal_places=1):
    if num == 0:
        return "0"
    elif num > 0.001 or num < -0.001:
        return num
    else:
        exponent = int(np.floor(np.log10(abs(num))))
        mantissa = num / 10**exponent
        return rf"${mantissa:.{decimal_places}f} \times 10^{{{exponent}}}$"


class Plot:
    """Plotting utilities for feature importance visualization."""
    
    # Constants
    TITLE_FONTSIZE = 20
    AXIS_LABEL_FONTSIZE = 25
    TICK_FONTSIZE = 15
    SCATTER_SIZE_LARGE = 450
    SCATTER_SIZE_MEDIUM = 200
    SCATTER_SIZE_SMALL = 70
    LINE_WIDTH = 2
    
    def __init__(self, config):
        self.config = config

    def plot_feature_importances(self, path, importances, name, sorting=None):
        """Plot feature importances as horizontal bar charts."""
        feature_names = self.config.dataset.feature_names

        if np.ndim(importances) == 1:
            importances = [importances]

        # Apply sorting if provided
        importances = [
            imp[sorting] if sorting is not None else imp
            for imp in importances
        ]
        feature_names = (
            [feature_names[i] for i in sorting] if sorting is not None else feature_names
        )

        n_plots = len(importances)
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
        
        if n_plots == 1:
            axes = [axes]

        for ax, importance in zip(axes, importances):
            self._plot_single_importance(ax, importance, feature_names)

        plt.tight_layout()
        plt.savefig(f"{path}/FI_{name}.png")
        plt.close()

    def _plot_single_importance(self, ax, importance, feature_names):
        """Plot a single importance bar chart."""
        df = pd.DataFrame({"Importance": importance}, index=feature_names)
        
        ax.barh(df.index, df["Importance"], color="b")
        ax.set_ylabel("Features", fontsize=self.TITLE_FONTSIZE)
        ax.set_xlabel("Importance", fontsize=self.TITLE_FONTSIZE)
        ax.tick_params(axis="both", which="major", labelsize=self.TICK_FONTSIZE)
        
        # Set symmetric x-limits
        max_abs_value = max(abs(np.min(df["Importance"])), abs(np.max(df["Importance"])))
        margin = 0.1 * max_abs_value
        ax.set_xlim(-max_abs_value - margin, max_abs_value + margin)

        # Add value labels on bars
        for y, (idx, row) in enumerate(df.iterrows()):
            self._add_bar_label(ax, row["Importance"], y, max_abs_value)

    def _add_bar_label(self, ax, value, y_pos, max_value):
        """Add label to a bar with intelligent positioning."""
        threshold = 0.5 * max_value
        
        if value == max_value:  # Positive maximum
            x_pos = value - 0.01 * max_value
            ha = "right"
            color = "white"
        elif -threshold < value < 0:  # Small negative values
            x_pos = 0.01 * max_value
            ha = "left"
            color = "black"
        else:  # Regular positive or large negative values
            x_pos = value + 0.01 * max_value
            ha = "left"
            color = "black"

        ax.text(
            x_pos, y_pos,
            f"{self._format_value(value)}",
            va="center", ha=ha, color=color,
            fontsize=self.TICK_FONTSIZE
        )

    @staticmethod
    def _format_value(value):
        """Format value with scientific notation."""
        return f"{sci_notation(round_to_second_significant(value))}"

    @staticmethod
    def calculate_colorpattern(tree_model, decision_tree, X_total, num_test_samples):
        """Calculate color patterns for tree leaves based on effective weights."""
        # Map leaves to their effective weight patterns
        leaf_patterns = {
            node.ID: tuple(node.effective_weights[1:])
            for node in tree_model
            if node.leaf
        }

        unique_patterns = list(set(leaf_patterns.values()))
        cmap = plt.get_cmap("tab10")

        # Create color mapping for each unique pattern
        pattern_to_color = {
            pattern: Plot._rgb_to_hex(cmap(i / len(unique_patterns)))
            for i, pattern in enumerate(unique_patterns)
        }

        # Assign patterns and colors to test samples
        point_patterns, point_colors = [], []
        for x in X_total[:num_test_samples]:
            leaf_node = [n for n in decision_tree.build_tree(x.reshape(1, -1)) if n.leaf][-1]
            pattern = tuple(leaf_node.effective_weights[1:])
            point_patterns.append(pattern)
            point_colors.append(pattern_to_color.get(pattern, "#000000"))

        return point_patterns, pattern_to_color

    @staticmethod
    def _rgb_to_hex(rgba):
        """Convert RGBA tuple to hex color string."""
        r, g, b, _ = rgba
        return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

class KrippendorffAlphaAnalysis:
    """Analysis of Krippendorff's alpha for inter-rater reliability."""
    
    # Constants
    FONTSIZE_LABEL = 20
    FONTSIZE_TITLE = 20
    FONTSIZE_SMALL = 15
    VIOLIN_COLOR = "#2196F3"
    VIOLIN_ALPHA = 0.7
    
    def __init__(self, config):
        self.config = config

    @staticmethod
    def _get_method_pairs(n_methods):
        """Generate all unique method pairs."""
        return list(combinations(range(n_methods), 2))

    @staticmethod
    def _generate_pair_labels(method_names):
        """Generate labels for method pairs."""
        pairs = KrippendorffAlphaAnalysis._get_method_pairs(len(method_names))
        return [f"{method_names[i]}\nvs\n{method_names[j]}" for i, j in pairs]

    def krippendorff_alpha_local(self, methods, level_of_measurement):
        """Compute Krippendorff's alpha between all method pairs across samples."""
        n_methods = len(methods)
        n_samples = len(methods[0])
        n_pairs = math.comb(n_methods, 2)
        
        alpha_values = np.empty((n_pairs, n_samples), dtype=float)
        pairs = self._get_method_pairs(n_methods)

        for pair_idx, (i, j) in enumerate(pairs):
            for sample_idx in range(n_samples):
                alpha_values[pair_idx, sample_idx] = self._compute_alpha_sample(
                    methods[i][sample_idx],
                    methods[j][sample_idx],
                    level_of_measurement
                )

        return alpha_values

    @staticmethod
    def _compute_alpha_sample(val_i, val_j, level_of_measurement):
        """Compute Krippendorff's alpha for a single sample pair."""
        data = np.array([val_i, val_j])
        try:
            return alpha(data, level_of_measurement=level_of_measurement)
        except Exception as e:
            print(f"Error computing alpha: {e}")
            return np.nan

    def krippendorff_alpha_global(self, data, level_of_measurement, missing_value=None):
        """Calculate Krippendorff's alpha for pairwise method comparisons."""
        n_methods = len(data)
        n_pairs = math.comb(n_methods, 2)
        alpha_values = np.empty(n_pairs, dtype=float)
        
        pairs = self._get_method_pairs(n_methods)
        for pair_idx, (i, j) in enumerate(pairs):
            pair_data = np.array([data[i], data[j]])
            alpha_values[pair_idx] = self._compute_alpha(
                pair_data, level_of_measurement, missing_value
            )

        return alpha_values

    @staticmethod
    def _compute_alpha(data, level_of_measurement, missing_value):
        """Calculate Krippendorff's alpha for a pair of raters."""
        data = np.array(data, dtype=float)
        
        if missing_value is not None:
            data[data == missing_value] = np.nan

        n_raters, n_items = data.shape
        all_values = data.flatten()
        valid_mask = ~np.isnan(all_values)
        
        if np.sum(valid_mask) < 2:
            return np.nan

        # Calculate observed disagreement
        observed_disagreement = KrippendorffAlphaAnalysis._calculate_disagreement(
            data, all_values[valid_mask], level_of_measurement, observed=True
        )
        
        if observed_disagreement is None:
            return np.nan

        # Calculate expected disagreement
        expected_disagreement = KrippendorffAlphaAnalysis._calculate_disagreement(
            None, all_values[valid_mask], level_of_measurement, observed=False
        )
        
        if expected_disagreement is None or expected_disagreement == 0:
            return 1.0 if expected_disagreement == 0 else np.nan

        alpha = 1 - (observed_disagreement / expected_disagreement)
        return alpha

    @staticmethod
    def _calculate_disagreement(data, valid_values, level_of_measurement, observed=True):
        """Calculate observed or expected disagreement."""
        metric_fn = KrippendorffAlphaAnalysis._get_metric_function(level_of_measurement, valid_values)
        
        if observed:
            return KrippendorffAlphaAnalysis._calculate_observed_disagreement(data, metric_fn)
        else:
            return KrippendorffAlphaAnalysis._calculate_expected_disagreement(valid_values, metric_fn)

    @staticmethod
    def _calculate_observed_disagreement(data, metric_fn):
        """Calculate observed disagreement across items."""
        disagreement = 0
        pair_count = 0

        n_raters, n_items = data.shape
        
        for item in range(n_items):
            item_values = data[:, item]
            valid_raters = ~np.isnan(item_values)
            
            if np.sum(valid_raters) < 2:
                continue

            valid_item_values = item_values[valid_raters]
            
            for val_i, val_j in combinations(valid_item_values, 2):
                disagreement += metric_fn(val_i, val_j)
                pair_count += 1

        return disagreement / pair_count if pair_count > 0 else None

    @staticmethod
    def _calculate_expected_disagreement(valid_values, metric_fn):
        """Calculate expected disagreement across all values."""
        disagreement = 0
        pair_count = 0

        for val_i, val_j in combinations(valid_values, 2):
            disagreement += metric_fn(val_i, val_j)
            pair_count += 1

        return disagreement / pair_count if pair_count > 0 else None

    @staticmethod
    def _get_metric_function(level_of_measurement, valid_values=None):
        """Get metric function based on level of measurement."""
        if level_of_measurement == 'interval':
            return lambda a, b: (a - b) ** 2
        elif level_of_measurement == 'ordinal':
            return KrippendorffAlphaAnalysis._create_ordinal_metric(valid_values)
        else:
            raise ValueError(f"Unsupported level of measurement: {level_of_measurement}")

    @staticmethod
    def _create_ordinal_metric(valid_values):
        """Create ordinal metric function."""
        unique_values, counts = np.unique(valid_values[~np.isnan(valid_values)], return_counts=True)
        
        def ordinal_metric(a, b):
            if a == b:
                return 0
            
            try:
                pos_a = np.where(unique_values == a)[0][0]
                pos_b = np.where(unique_values == b)[0][0]
            except IndexError:
                return 0
            
            min_pos, max_pos = min(pos_a, pos_b), max(pos_a, pos_b)
            
            sum_between = sum(
                counts[i] / 2 if i in [min_pos, max_pos] else counts[i]
                for i in range(min_pos, max_pos + 1)
            )
            
            return sum_between ** 2
        
        return ordinal_metric

    def plot_krippendorff_violin(self, method_names, method_results, save_path=None, scale=0):
        """ Plot violin plots of Krippendorff's alpha values. """
        pair_labels = self._generate_pair_labels(method_names)
        n_pairs = len(pair_labels)

        if n_pairs == 0:
            print("No valid alpha data to plot.")
            return

        fig, axes = plt.subplots(1, n_pairs, figsize=(3 * n_pairs, 5), sharey=False)
        
        if n_pairs == 1:
            axes = [axes]

        y_min, y_max = self._calculate_y_limits(method_results, scale)

        for ax, data, label in zip(axes, method_results, pair_labels):
            self._plot_single_violin(ax, data, label, y_min, y_max, scale)

        plt.tight_layout()
        self._save_figure(save_path)

    @staticmethod
    def _calculate_y_limits(method_results, scale):
        """Calculate y-axis limits based on scale mode."""
        if scale == 1:
            all_data = np.concatenate(method_results)
            global_min = np.min(all_data)
            global_max = np.max(all_data)
            padding = (global_max - global_min) * 0.1
            return global_min - padding, global_max + padding
        else:
            return None, None

    def _plot_single_violin(self, ax, data, label, global_min, global_max, scale):
        """Plot a single violin plot."""
        if scale == 0:
            y_min = np.min(data) - (np.max(data) - np.min(data)) * 0.1
            y_max = np.max(data) + (np.max(data) - np.min(data)) * 0.1
        else:
            y_min, y_max = global_min, global_max

        violin_parts = ax.violinplot(data, positions=[1], showmeans=True, showextrema=True)
        
        # Style violin plot
        violin_parts["bodies"][0].set_facecolor(self.VIOLIN_COLOR)
        violin_parts["bodies"][0].set_alpha(self.VIOLIN_ALPHA)
        violin_parts["cmeans"].set_color("red")
        violin_parts["cmeans"].set_linewidth(2)

        for partname in ["cbars", "cmins", "cmaxes"]:
            violin_parts[partname].set_color("black")
            violin_parts[partname].set_linewidth(1.5)

        # Configure axes
        ax.set_xticks([1])
        ax.set_xticklabels([label], fontsize=self.FONTSIZE_LABEL)
        ax.set_ylim(y_min, y_max)
        ax.tick_params(axis="y", labelsize=self.FONTSIZE_LABEL)
        ax.grid(True, linestyle="--", alpha=0.7, axis="y")
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(style="sci", axis="y", useMathText=True)
        ax.yaxis.get_offset_text().set_fontsize(self.FONTSIZE_SMALL)

        # Label only first subplot
        if ax == plt.gca().figure.axes[0]:
            ax.set_ylabel("Krippendorff's Alpha", fontsize=self.FONTSIZE_TITLE)

    @staticmethod
    def _save_figure(save_path):
        """Save figure to file."""
        filename = save_path if save_path else "Krippendorff_comparison_pairs_violin.png"
        plt.savefig(filename, bbox_inches="tight", dpi=300)
        plt.close()


class RMSEAnalysis:
    """RMSE analysis between XAI methods."""
    
    def __init__(self, config):
        self.config = config

    @staticmethod
    def _calculate_rmse_pair(result1, result2):
        """Calculate RMSE between two methods."""
        result1 = np.array(result1)
        result2 = np.array(result2)
        
        if result1.ndim == 1:
            result1 = result1.reshape(-1, 1)
            result2 = result2.reshape(-1, 1)
            
        return np.sqrt(np.mean((result1 - result2) ** 2, axis=1))

    def calculate_rmse(self, method_names, method_results, save_path=None, plot=False):
        """Calculate RMSE statistics between all method pairs."""
        rmse_stats = {}

        for i, j in combinations(range(len(method_names)), 2):
            pair_name = f"{method_names[i]} vs {method_names[j]}"
            rmse_samples = self._calculate_rmse_pair(method_results[i], method_results[j])
            
            rmse_stats[pair_name] = {
                "mean": np.mean(rmse_samples),
                "median": np.median(rmse_samples),
                "std": np.std(rmse_samples),
                "min": np.min(rmse_samples),
                "max": np.max(rmse_samples),
            }

        if plot:
            self.plot_rmse_violin(method_names, method_results, save_path=save_path)
        
        return rmse_stats

    def save_rmse_stats(self, rmse_stats, save_path):
        """Save RMSE statistics to file."""
        if isinstance(rmse_stats, list):
            rmse_stats = rmse_stats[0]
        
        with open(f"{save_path}.csv", "w") as f:
            for pair, stats in rmse_stats.items():
                f.write(f"{pair}:\n")
                f.write(f"  Mean RMSE: {stats['mean']}\n")
                f.write(f"  Median RMSE: {stats['median']}\n")
                f.write(f"  Std Dev: {stats['std']}\n")
                f.write(f"  Range: [{stats['min']}, {stats['max']}]\n\n")

    def plot_rmse_violin(self, method_names, method_results, percentile_threshold=95, save_path=None):
        """Plot violin plots of RMSE between method pairs."""
        violin_data = []
        pair_labels = []

        for i, j in combinations(range(len(method_names)), 2):
            rmse = self._calculate_rmse_pair(method_results[i], method_results[j])
            threshold = np.percentile(rmse, percentile_threshold)
            filtered_rmse = rmse[rmse <= threshold]
            
            violin_data.append(filtered_rmse)
            pair_labels.append(f"{method_names[i]}\nvs\n{method_names[j]}")

        n_pairs = len(pair_labels)
        if n_pairs == 0:
            return
        
        fig, axes = plt.subplots(1, n_pairs, figsize=(3 * n_pairs, 5), sharey=False)
        if n_pairs == 1:
            axes = [axes]

        # Calculate global y-limits
        all_data = np.concatenate(violin_data)
        global_min = np.min(all_data)
        global_max = np.max(all_data)
        padding = (global_max - global_min) * 0.1

        for idx, (ax, data, label) in enumerate(zip(axes, violin_data, pair_labels)):
            self._plot_single_violin(ax, data, label, global_min, global_max, padding, idx == 0)

        plt.tight_layout()
        filename = f"{save_path}.png" if save_path else "rmse_comparison_pairs_violin.png"
        plt.savefig(filename, bbox_inches="tight", dpi=300)
        plt.close()

    @staticmethod
    def _plot_single_violin(ax, data, label, global_min, global_max, padding, is_first):
        """Plot a single violin plot."""
        # Calculate local y-limits
        local_min = np.min(data) - (np.max(data) - np.min(data)) * 0.1
        local_max = np.max(data) + (np.max(data) - np.min(data)) * 0.1

        v = ax.violinplot(data, positions=[1], showmeans=True, showextrema=True)

        # Style violin
        v["bodies"][0].set_facecolor("#2196F3")
        v["bodies"][0].set_alpha(0.7)
        v["cmeans"].set_color("red")
        v["cmeans"].set_linewidth(2)

        for partname in ["cbars", "cmins", "cmaxes"]:
            v[partname].set_color("black")
            v[partname].set_linewidth(1.5)

        # Configure axes
        ax.set_xticks([1])
        ax.set_xticklabels([label], fontsize=20)
        ax.set_ylim(local_min, local_max)
        ax.tick_params(axis="y", labelsize=20)
        ax.grid(True, linestyle="--", alpha=0.7, axis="y")
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(style="sci", axis="y", useMathText=True)
        ax.yaxis.get_offset_text().set_fontsize(15)

        if is_first:
            ax.set_ylabel("Mean Squared Error", fontsize=20)