from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error

from src.utils import TaskType


class Net(nn.Module):
    def __init__(self, layers, task_type, num_output):
        super().__init__()
        self.task_type = task_type
        self.num_output = num_output
        
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

    def forward(self, x):
        # Hidden layers with ReLU
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.relu(x)
        
        # Output layer
        x = self.layers[-1](x)
        
        # Output activation based on task type
        if self.task_type == TaskType.CLASSIFICATION:
            x = F.softmax(x, dim=1)
        # For regression, linear activation (no activation)
        
        return x


class DNN:
    def __init__(self, task_type, num_input, hidden_nodes, num_output, directory, timestamp):
        self.task_type = task_type
        self.num_input = num_input
        self.hidden_nodes = hidden_nodes
        self.num_output = num_output
        self.directory = Path(directory)
        self.timestamp = timestamp
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def create_model(self, load_existing, model_filename=None, X_train=None, X_test=None, 
                     y_train=None, y_test=None):
        """Load or train neural network model."""
        if load_existing:
            self.model = self._load_model(model_filename)
        else:
            self.model = self._build_and_train_model(X_train, X_test, y_train, y_test)
            self._save_model()
        return self.model

    def _build_and_train_model(self, X_train, X_test, y_train, y_test):
        """Build and train neural network model."""
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        # Build architecture
        layers = [self.num_input] + self.hidden_nodes + [self.num_output]
        self.model = Net(layers, self.task_type, self.num_output).to(self.device)
        
        # Configure output layer and compile
        if self.task_type == TaskType.REGRESSION:
            y_train_tensor = torch.FloatTensor(y_train).to(self.device)
            y_test_tensor = torch.FloatTensor(y_test).to(self.device)
            if len(y_train_tensor.shape) == 1:
                y_train_tensor = y_train_tensor.unsqueeze(1)
                y_test_tensor = y_test_tensor.unsqueeze(1)
            
            loss_function = nn.MSELoss()
            monitor_metric = 'val_loss'
            patience = 20
            epochs = 3000
            batch_size = None
        else:  # CLASSIFICATION
            y_train_tensor = torch.LongTensor(y_train).to(self.device)
            y_test_tensor = torch.LongTensor(y_test).to(self.device)
            
            loss_function = nn.CrossEntropyLoss()
            monitor_metric = 'val_accuracy'
            patience = 10
            epochs = 200
            batch_size = 8
        
        # Train model
        self._train_model(
            X_train_tensor, y_train_tensor,
            X_test_tensor, y_test_tensor,
            loss_function, epochs, batch_size, patience, monitor_metric
        )
        
        return self.model

    def _train_model(self, X_train, y_train, X_test, y_test, 
                     loss_function, epochs, batch_size, patience, monitor_metric):
        """Train model with early stopping."""
        optimizer = optim.Adam(self.model.parameters())
        
        best_val_loss = float('inf')
        best_val_acc = 0.0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            self.model.train()
            
            # Training
            if batch_size:
                # Batch training for classification
                indices = torch.randperm(X_train.size(0))
                for i in range(0, X_train.size(0), batch_size):
                    batch_indices = indices[i:i+batch_size]
                    X_batch = X_train[batch_indices]
                    y_batch = y_train[batch_indices]
                    
                    optimizer.zero_grad()
                    y_pred = self.model(X_batch)
                    loss = loss_function(y_pred, y_batch)
                    loss.backward()
                    optimizer.step()
            else:
                # Full batch for regression
                optimizer.zero_grad()
                y_pred = self.model(X_train)
                loss = loss_function(y_pred, y_train)
                loss.backward()
                optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_test)
                val_loss = loss_function(val_pred, y_test).item()
                
                if monitor_metric == 'val_loss':
                    # Early stopping based on validation loss
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_model_state = self.model.state_dict().copy()
                    else:
                        patience_counter += 1
                else:  # 'val_accuracy'
                    # Calculate accuracy
                    pred_labels = torch.argmax(val_pred, dim=1)
                    val_acc = (pred_labels == y_test).float().mean().item()
                    
                    # Early stopping based on validation accuracy
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        patience_counter = 0
                        best_model_state = self.model.state_dict().copy()
                    else:
                        patience_counter += 1
            
            # Check early stopping
            if patience_counter >= patience:
                break
        
        # Restore best weights
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

    def _load_model(self, filename):
        """Load existing model from file."""
        model_path = self.directory / filename
        print(f"Loading model from {model_path}")
        
        # Reconstruct model architecture
        layers = [self.num_input] + self.hidden_nodes + [self.num_output]
        model = Net(layers, self.task_type, self.num_output).to(self.device)
        
        # Load state dict
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        
        return model

    def _save_model(self):
        """Save trained model to file."""
        model_dir = self.directory
        model_dir.mkdir(parents=True, exist_ok=True)
        
        nodes_str = "-".join(map(str, self.hidden_nodes))
        time_str = self.timestamp.strftime("%Y%m%d-%H%M")
        filename = model_dir / f"DNNmodel_{nodes_str}_{time_str}.pt"
        
        torch.save(self.model.state_dict(), filename)

    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        self.model.eval()
        
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_test_tensor).cpu().numpy()
        
        if self.task_type == TaskType.REGRESSION:
            mse = mean_squared_error(y_test, predictions)
            return mse
        
        pred_labels = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(y_test, pred_labels)
        return accuracy
    
    def load_weight_bias(self):
        """Load weights and biases from all layers."""
        weight = []
        bias = []

        for layer in self.model.layers:
            # Transpose to match TensorFlow format: (in_features, out_features)
            weight.append(layer.weight.data.cpu().numpy().T)
            bias.append(layer.bias.data.cpu().numpy())

        return (weight, bias)

    def load_layer_info(self):
        """Load layer information including weights, biases, and neuron counts."""
        layers = []
        
        for layer in self.model.layers:
            if isinstance(layer, nn.Linear):
                layer_dict = {"layer": "Dense"}
                
                # Transpose weights to match TensorFlow format: (in_features, out_features)
                layer_dict["weights"] = layer.weight.data.cpu().numpy().T
                layer_dict["bias"] = layer.bias.data.cpu().numpy()
                layer_dict["neurons"] = layer.out_features
                
                layers.append(layer_dict)
        
        return layers

    def get_layer_activations(self, input_data):
        """Get activations from all layers for given input."""
        self.model.eval()
        
        input_reshaped = input_data.reshape(1, -1)
        input_tensor = torch.FloatTensor(input_reshaped).to(self.device)
        
        activations = []
        x = input_tensor
        
        with torch.no_grad():
            for i, layer in enumerate(self.model.layers):
                x = layer(x)
                if i < len(self.model.layers) - 1:
                    # ReLU for hidden layers
                    x = F.relu(x)
                elif self.task_type == TaskType.CLASSIFICATION:
                    # Softmax for classification output
                    x = F.softmax(x, dim=1)
                
                activations.append(x.cpu().numpy())
        
        return activations
    
    def predict(self, X):
        """Make predictions using the trained model."""
        self.model.eval()

        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        if self.task_type == TaskType.CLASSIFICATION:
            predictions = np.argmax(predictions, axis=1)
        elif self.task_type == TaskType.REGRESSION and self.num_output == 1:
            predictions = predictions.flatten()
        
        return predictions