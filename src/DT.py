import pickle
from typing import List, Optional, Union

import numpy as np

from src.utils import Framework, TaskType, calculate_acc, calculate_mse


class Node:
    """Represents a node in the decision tree."""

    def __init__(
        self,
        col: int,
        node_id: int,
        is_leaf: bool,
        neuron_id: List[int],
        num_samples: Optional[int] = None,
        branch: Optional[int] = None,
        activation: Optional[float] = None,
        activation_pattern: Optional[np.ndarray] = None,
        memory: Optional[List[np.ndarray]] = None,
        parent_id: Optional[int] = None,
        effective_weights: Optional[np.ndarray] = None,
    ):
        self.col = col  # stage number in DT
        self.parent_id = parent_id  # node_id of parent node --> link
        self.node_id = node_id  # identification --> every node has an individual node_id
        self.branch = branch  # branch value (for binary: True=1, False=0)
        self.activation = activation  # activation (with ReLU activation = branch value)
        self.activation_pattern = activation_pattern  # activation pattern
        self.num_samples = (
            num_samples  # number of training samples flow through that is_leaf
        )
        self.memory = memory  # store samples fall into that is_leaf
        self.is_leaf = is_leaf  # FALSE for nodes, TRUE for leave nodes
        self.neuron_id = (
            neuron_id  # [hidden_layer, neuron, number of neurons in previous layer]
        )
        self.effective_weights = effective_weights # weights and bias considering activation pattern


class BaseTree:    
    """Base class for Decision Trees."""
    def __init__(
        self, 
        layers: List[dict], 
        activation_function: str, 
        num_input: int, 
        task_type: TaskType, 
        num_output: int, 
        framework: Framework
    ):
        self.layers = layers
        self.activation_function = activation_function
        self.num_input = num_input
        self.task_type = task_type
        self.num_output = num_output
        self.framework = framework

    # ============================================================================
    # NODE CREATION
    # ============================================================================


    def _create_node(
        self,
        col: int,
        neuron_id: List[int],
        node_id: int,
        is_leaf: bool,
        num_samples: Optional[int] = None,
        branch: Optional[int] = None,
        parent_id: Optional[int] = None,
        parent_activation_pattern: Optional[np.ndarray] = None,
    ):
        
        if self.activation_function == "relu":
            activation = branch
        else:
            raise ValueError("Unsupported activation function.")
        
        if parent_activation_pattern is None:
            activation_pattern = [branch] if branch is not None else None
        else:
            activation_pattern = parent_activation_pattern + [branch]

        return Node(
                col=col,
                neuron_id=neuron_id,
                num_samples=num_samples,
                parent_id=parent_id,
                node_id=node_id,
                branch=branch,
                activation=activation,
                activation_pattern=activation_pattern,
                is_leaf=is_leaf,
            )

    def _find_node_by_id(self, tree, node_id):
        for node in reversed(tree):
            if node.node_id == node_id:
                return node
        return None
    
    # ============================================================================
    # WEIGHT MATRIX OPERATIONS
    # ============================================================================
    
    def _combine_weights_bias(self, layer_idx: int):
        layer = self.layers[layer_idx]
        # get layer weights
        weight_matrix = layer['weights'].T
        # add bias in first column
        weight_matrix = np.insert(weight_matrix, 0, layer['bias'], axis=1)
        # create first row for bias pass-through
        first_row = np.zeros(np.shape(weight_matrix)[1])
        first_row[0] = 1
        weight_matrix = np.vstack([first_row, weight_matrix])
        return weight_matrix

    def _apply_activation_mask(self, layer_idx: int, activation: np.ndarray):
        if self.layers[layer_idx]['layer'] == 'Dense':
            weight_matrix = self._combine_weights_bias(layer_idx)
            # insert True for bias row
            activation = np.insert(activation, 0, True)
            # set weights to 0 for deactivated neurons
            weight_matrix[~activation] = 0
            return weight_matrix
        else:
            raise ValueError("Error in calculating effective weights")
    
    def _generate_weight_matrix(self, layer_idx: int):
        """ Generate weight matrix with bias for a given layer type."""
        weight_matrix = []
        if self.layers[layer_idx]['layer'] == 'Dense':
            weight_matrix = self._combine_weights_bias(layer_idx)
            return weight_matrix
        else:
            raise ValueError("Error in calculating weights")


    def _calculate_effective_weight_matrix(self, layer_idx: int, neuron_idx: int, decisions_per_layer: List[int], node, is_leaf_node: bool):
        if layer_idx == 0:
            # first hidden layer: use direct weights and bias
            weight_matrix = self._generate_weight_matrix(layer_idx)
            node.effective_weights = weight_matrix[neuron_idx+1]
        else:
            # Compute activation pattern from parent path
            activation_pattern = np.array(node.activation_pattern, dtype=bool)

            # Initialize cumulative weight
            activation_matrix = np.eye(self.num_input + 1)

            # Compute cumulative transformation from input to current layer
            for i in range(layer_idx):
                start_idx = sum(decisions_per_layer[:i])
                end_idx = sum(decisions_per_layer[: i + 1])
                # extract activation pattern for layer i
                layer_activation = activation_pattern[start_idx : end_idx]
                # apply activation mask to weights
                activated_weight = self._apply_activation_mask(i, layer_activation)
                # update cumulative activation matrix
                activation_matrix = np.dot(activated_weight, activation_matrix)

            # get weight matrix for current layer
            weight_matrix = self._generate_weight_matrix(layer_idx)
            # combine with cumulative activation matrix to get effective weight matrix
            effective_weight_matrix = np.dot(weight_matrix, activation_matrix)
                
            if is_leaf_node:
                # Remove bias row (first row)
                node.effective_weights = effective_weight_matrix[1:]
                # For regression with single output, flatten to 1D
                if self.task_type == TaskType.REGRESSION and self.num_output == 1:
                    node.effective_weights = node.effective_weights.flatten()
            else:
                node.effective_weights = effective_weight_matrix[neuron_idx+1]
    
    def _calculate_number_of_decisions(self, layer: dict):
        decisions = 0
        if layer['layer'] == "Dense":
            # number of neurons are the number of decision points
            decisions = layer['neurons']
        return int(decisions)
    
    # ============================================================================
    # EVALUATION
    # ============================================================================

    def evaluate(self, X_test, y_test):
        tree_predictions = self.predict(X_test)
        num_samples_test = X_test.shape[0]

        if self.task_type == TaskType.REGRESSION:
            err_tree = calculate_mse(tree_predictions, y_test, num_samples_test)
        elif self.task_type == TaskType.CLASSIFICATION:
            err_tree = calculate_acc(tree_predictions, y_test, num_samples_test)
        else:
            raise ValueError("Invalid task definition.")

        return err_tree
    
    def save_tree(self, tree, filename: str):
        """Save the decision tree to a file."""
        with open(filename, 'wb') as f:
            pickle.dump(tree, f)

    def load_tree(self, filename: str):
        """Load the decision tree from a file."""
        with open(filename, 'rb') as f:
            tree = pickle.load(f)
            self.tree = tree
        return tree
    

class Tree(BaseTree):
    """
    Neural Network-based Decision Tree.
    
    Constructs a decision tree from a trained neural network using a depth-first strategy.
    Each training sample grows a unique path from root to leaf, creating a compact tree.
    """
    

    # ============================================================================
    # DATA PREPROCESSING
    # ============================================================================
    
    def _preprocess_data(self, data):
        # Insert 1 for bias pass-through
        if data.ndim == 1:
            data = np.insert(data, 0, 1)
        elif data.ndim > 2:
            data = [sample.flatten() for sample in data]
            data = np.insert(data, 0, 1, axis=1)
        else: 
            data = np.insert(data, 0, 1, axis=1)
        return data

    # ============================================================================
    # TREE BUILDING
    # ============================================================================

    def build_tree(self, data):
        """
        build tree with commmon oblique decision tree structure as in
        scikit-obliquetree https://github.com/hengzhe-zhang/scikit-obliquetree
        """
        tree, current_parents = [], []
        dnn_node_id = 0
        total_samples = len(data)

        data = self._preprocess_data(data)

        # calculate the number of decisions per layer / number of level in the DT
        decisions_per_layer = [self._calculate_number_of_decisions(layer) for layer in self.layers[:-1]]
               
        for layer_idx, _ in enumerate(self.layers[:-1]):
            for neuron_idx in range(decisions_per_layer[layer_idx]):
                number_of_decisions = None if layer_idx == 0 else self._calculate_number_of_decisions(self.layers[layer_idx - 1])

                if layer_idx == 0 and neuron_idx == 0:
                    # Root node
                    node = self._create_node(col=dnn_node_id,
                                             neuron_id=[layer_idx, neuron_idx, number_of_decisions],
                                             num_samples=total_samples,
                                             node_id=0,
                                             is_leaf=False)       
                        #dnn_node_id, [layer_idx, neuron_idx, number_of_decisions], total_samples)
                    self._calculate_effective_weight_matrix(layer_idx, neuron_idx, decisions_per_layer, node, False)
                    node.memory = data
                    tree.append(node)
                    current_parents.append(node.node_id)
                    dnn_node_id += 1
                else:
                    current_children = []
                    for parent_id in current_parents:
                        parent_node = self._find_node_by_id(tree, parent_id)
                        parent_activation_pattern = parent_node.activation_pattern if parent_node else None

                        for branch in [1, 0]:
                            node_id = parent_id * 2 + (1 if branch else 2)
                            node = self._create_node(
                                col=dnn_node_id,
                                neuron_id=[layer_idx, neuron_idx, number_of_decisions],
                                parent_id=parent_id,
                                parent_activation_pattern=parent_activation_pattern,
                                node_id=node_id,
                                branch=branch,
                                is_leaf=False,
                            )
                            self._calculate_effective_weight_matrix(
                                layer_idx, neuron_idx, decisions_per_layer, node, False
                            )

                            weighted_sum = parent_node.memory.dot(
                                    parent_node.effective_weights
                                )

                            child_samples = (
                                parent_node.memory[weighted_sum >= 0]
                                if branch
                                else parent_node.memory[weighted_sum < 0]
                            )

                            node.memory = child_samples
                            node.num_samples = child_samples.shape[0]

                            if node.num_samples > 0:
                                current_children.append(node.node_id)
                                tree.append(node)

                    current_parents = current_children
                    dnn_node_id += 1

        self._leaf_nodes(tree, current_parents, decisions_per_layer)
        return tree

    def _leaf_nodes(self, tree, current_parents, decisions_per_layer):
        for parent_id in current_parents:
            parent_node = self._find_node_by_id(tree, parent_id)
            parent_activation_pattern = parent_node.activation_pattern

            for branch in [1, 0]:
                node_id = parent_id * 2 + (1 if branch else 2)
                node = self._create_node(
                    sum(decisions_per_layer),
                    [len(decisions_per_layer), 0, decisions_per_layer[-1]],
                    parent_id=parent_id,
                    parent_activation_pattern=parent_activation_pattern,
                    node_id=node_id,
                    branch=branch,
                    is_leaf=True,
                )

                self._calculate_effective_weight_matrix(len(self.layers) - 1, None, decisions_per_layer, node, True)

                # Assign memory to is_leaf
                weighted_sums = parent_node.memory.dot(parent_node.effective_weights)
                    
                node.memory = (
                    parent_node.memory[weighted_sums >= 0]
                    if branch
                    else parent_node.memory[weighted_sums < 0]
                )
                node.num_samples = node.memory.shape[0]

                if node.num_samples > 0:
                    tree.append(node)

    # ============================================================================
    # PREDICTION
    # ============================================================================

    def _predict_sample(self, sample, tree):
        leaf_node = tree[-1]
        sample = np.insert(sample, 0, 1)
        if self.task_type == TaskType.REGRESSION:
            return (
                np.dot(sample, leaf_node.effective_weights)
            )
        else:
            output = []
            for class_weights in leaf_node.effective_weights:
                class_output = np.dot(sample, class_weights) 
                output.append(class_output)
            return np.array(output)

    def predict(self, data, tree=None):
        predictions = []
        for sample in data:
            sample = np.array([[sample]]) if np.isscalar(sample) else sample
            sample = sample.flatten() if sample.ndim > 1 else sample
            tree = self.build_tree(sample)
            prediction = self._predict_sample(sample, tree)
            predictions.append(prediction)

        predictions = np.array(predictions)
        if self.task_type == TaskType.REGRESSION:
            return predictions
        elif self.task_type == TaskType.CLASSIFICATION:
            return np.argmax(predictions, axis=1)
        else:
            raise ValueError("Invalid task_type value")

class CompleteTree(BaseTree):
    """
    Complete Neural Network-based Decision Tree.
    
    Constructs a complete decision tree from a trained neural network using a breadth-first strategy.
    The entire tree is pre-built based on network structure, allowing direct evaluation of any data point.
    Trade-off: Larger memory footprint for faster inference without reconstruction.
    """

    # ============================================================================
    # TREE BUILDING
    # ============================================================================

    def build_tree(self, X_train: np.ndarray):
        """
        Build complete tree structure using breadth-first strategy.
        All possible paths are pre-constructed based on network architecture.
        """
        tree = []
        current_parents = []
        dnn_node_id = 0

        # Preprocess training data
        X_train_processed = self._preprocess_data(X_train)
        total_samples = len(X_train)

        # Calculate number of decisions per layer
        decisions_per_layer = [
            self._calculate_number_of_decisions(layer) 
            for layer in self.layers[:-1]
        ]

        # Build tree structure layer by layer
        for layer_idx, _ in enumerate(self.layers[:-1]):
            for neuron_idx in range(decisions_per_layer[layer_idx]):
                number_of_decisions = (
                    None if layer_idx == 0 
                    else self._calculate_number_of_decisions(self.layers[layer_idx - 1])
                )

                if layer_idx == 0 and neuron_idx == 0:
                    # Root node
                    node = self._create_node(
                        col=dnn_node_id,
                        neuron_id=[layer_idx, neuron_idx, number_of_decisions],
                        node_id=0,
                        num_samples=total_samples,
                        is_leaf=False,
                    )
                    self._calculate_effective_weight_matrix(
                        layer_idx, neuron_idx, decisions_per_layer, node, False
                    )
                    node.memory = X_train_processed
                    tree.append(node)
                    current_parents.append(node.node_id)
                    dnn_node_id += 1
                else:
                    current_children = []
                    for parent_id in current_parents:
                        parent_node = self._find_node_by_id(tree, parent_id)
                        parent_activation_pattern = (
                            parent_node.activation_pattern if parent_node else None
                        )

                        # Create both branches (True and False)
                        for branch in [1, 0]:
                            node_id = parent_id * 2 + (1 if branch else 2)
                            node = self._create_node(
                                col=dnn_node_id,
                                neuron_id=[layer_idx, neuron_idx, number_of_decisions],
                                parent_id=parent_id,
                                parent_activation_pattern=parent_activation_pattern,
                                node_id=node_id,
                                branch=branch,
                                is_leaf=False,
                            )
                            self._calculate_effective_weight_matrix(
                                layer_idx, neuron_idx, decisions_per_layer, node, False
                            )

                            # Calculate which samples flow through this node
                            weighted_sum = parent_node.memory.dot(
                                parent_node.effective_weights
                            )
                            child_samples = (
                                parent_node.memory[weighted_sum >= 0]
                                if branch
                                else parent_node.memory[weighted_sum < 0]
                            )

                            node.memory = child_samples
                            node.num_samples = child_samples.shape[0]

                            tree.append(node)
                            current_children.append(node.node_id)

                    current_parents = current_children
                    dnn_node_id += 1

        # Create leaf nodes
        self._create_leaf_nodes(tree, current_parents, decisions_per_layer)
        self.tree = tree
        return tree

    def _create_leaf_nodes(self, tree, current_parents, decisions_per_layer):
        """Create leaf nodes for all terminal branches."""
        for parent_id in current_parents:
            parent_node = self._find_node_by_id(tree, parent_id)
            parent_activation_pattern = parent_node.activation_pattern

            for branch in [1, 0]:
                child_id = parent_id * 2 + (1 if branch else 2)
                node = self._create_node(
                    col=sum(decisions_per_layer),
                    neuron_id=[len(decisions_per_layer), 0, decisions_per_layer[-1]],
                    parent_id=parent_id,
                    parent_activation_pattern=parent_activation_pattern,
                    node_id=child_id,
                    branch=branch,
                    is_leaf=True,
                )

                self._calculate_effective_weight_matrix(
                    len(self.layers) - 1, None, decisions_per_layer, node, True
                )

                # Assign memory to leaf
                weighted_sums = parent_node.memory.dot(parent_node.effective_weights)
                node.memory = (
                    parent_node.memory[weighted_sums >= 0]
                    if branch
                    else parent_node.memory[weighted_sums < 0]
                )
                node.num_samples = node.memory.shape[0]

                tree.append(node)

    # ============================================================================
    # DATA PREPROCESSING
    # ============================================================================
    
    def _preprocess_data(self, data):
        """Preprocess data by adding bias term."""
        # Insert 1 for bias pass-through
        if data.ndim == 1:
            data = np.insert(data, 0, 1)
        elif data.ndim > 2:
            data = [sample.flatten() for sample in data]
            data = np.insert(data, 0, 1, axis=1)
        else: 
            data = np.insert(data, 0, 1, axis=1)
        return data

    # ============================================================================
    # PREDICTION
    # ============================================================================

    def _predict_sample(self, sample, tree=None):
        """Predict for a single sample using the complete tree."""
        if self.tree is None:
            raise ValueError("Tree not built. Call build_tree() first.")

        # Navigate tree based on weighted sums
        current_node = self.tree[0]
        sample_with_bias = np.insert(sample, 0, 1)

        while not current_node.is_leaf:
            weighted_sum = np.dot(
                sample_with_bias, 
                current_node.effective_weights
            )
            next_id = current_node.node_id * 2 + (1 if weighted_sum >= 0 else 2)
            current_node = next(
                node for node in self.tree if node.node_id == next_id
            )

        # Calculate output at leaf node
        if self.task_type == TaskType.REGRESSION:
            return np.dot(sample_with_bias, current_node.effective_weights)
        else:  # Classification
            output = []
            for class_weights in current_node.effective_weights:
                class_output = np.dot(sample_with_bias, class_weights)
                output.append(class_output)
            return np.array(output)

    def predict(self, data):
        """Predict for multiple samples."""
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if len(data.shape) == 1:
            data = data.reshape(1, -1)

        predictions = []
        for sample in data:
            sample = sample.flatten() if sample.ndim > 1 else sample
            prediction = self._predict_sample(sample)
            predictions.append(prediction)

        predictions = np.array(predictions)
        
        if self.task_type == TaskType.REGRESSION:
            return predictions
        elif self.task_type == TaskType.CLASSIFICATION:
            return np.argmax(predictions, axis=1)
        else:
            raise ValueError("Invalid task_type value")