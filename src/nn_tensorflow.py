from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import accuracy_score, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from src.utils import TaskType


class DNN:
    def __init__(self, task_type, num_input, hidden_nodes, num_output, directory, timestamp):
        self.task_type = task_type
        self.num_input = num_input
        self.hidden_nodes = hidden_nodes
        self.num_output = num_output
        self.directory = Path(directory)
        self.timestamp = timestamp
        self.model = None

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
        self.model = Sequential()
        
        # Build architecture
        self.model.add(Dense(self.hidden_nodes[0], activation='relu', input_dim=self.num_input))
        for units in self.hidden_nodes[1:]:
            self.model.add(Dense(units, activation='relu'))
        
        # Configure output layer and compile
        if self.task_type == TaskType.REGRESSION:
            self.model.add(Dense(self.num_output, activation='linear'))
            self.model.compile(optimizer='adam', loss='mse', metrics=['mse'])
            y_train_processed, y_test_processed = y_train, y_test
            callback = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
            epochs = 3000
            batch_size = None
        else:
            self.model.add(Dense(self.num_output, activation='softmax'))
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            y_train_processed = tf.keras.utils.to_categorical(y_train, num_classes=self.num_output)
            y_test_processed = tf.keras.utils.to_categorical(y_test, num_classes=self.num_output)
            callback = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
            epochs = 200
            batch_size = 8
        
        # Train model
        fit_kwargs = {'epochs': epochs, 'validation_data': (X_test, y_test_processed), 
                      'callbacks': [callback]}
        if batch_size:
            fit_kwargs['batch_size'] = batch_size
            
        self.model.fit(X_train, y_train_processed, **fit_kwargs)
        return self.model

    def _load_model(self, filename):
        """Load existing model from file."""
        model_path = self.directory / filename
        print(f"Loading model from {model_path}")
        model = load_model(model_path, compile=False)
        loss = 'mse' if self.task_type == TaskType.REGRESSION else 'categorical_crossentropy'
        model.compile(optimizer="adam", loss=loss)
        return model

    def _save_model(self):
        """Save trained model to file."""
        model_dir = self.directory
        model_dir.mkdir(parents=True, exist_ok=True)
        
        nodes_str = "-".join(map(str, self.hidden_nodes))
        time_str = self.timestamp.strftime("%Y%m%d-%H%M")
        filename = model_dir / f"DNNmodel_{nodes_str}_{time_str}.h5"
        
        self.model.save(filename)

    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        predictions = self.model.predict(X_test)
        
        if self.task_type == TaskType.REGRESSION:
            mse = mean_squared_error(y_test, predictions)
            return mse
        
        pred_labels = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(y_test, pred_labels)
        return accuracy
    
    def load_weight_bias(self):

        weight = []
        bias = []

        for i in range(len(self.model.layers)):
            weight.append(self.model.layers[i].get_weights()[0])
            bias.append(self.model.layers[i].get_weights()[1])

        return (weight, bias)
    

    def load_layer_info(self):
        layers = []
        for layer in self.model.layers:
            if layer.__class__.__name__ != 'Flatten':
                layer_dict = {"layer": layer.__class__.__name__}
                weights = layer.get_weights()
                if weights:
                    layer_dict["weights"] = weights[0]
                    layer_dict["bias"] = weights[1]

                layer_dict["neurons"] = layer.units
                
                layers.append(layer_dict)
        
        return layers


    def get_layer_activations(self, input_data):
        """Get activations from all layers for given input."""
        input_reshaped = input_data.reshape(1, -1)
        activations = []
        
        for layer in self.model.layers:
            activation_func = tf.keras.backend.function([self.model.input], [layer.output])
            activations.append(activation_func([input_reshaped])[0])
        
        return activations
    
    def predict(self, X):
        """Make predictions using the trained model."""
        predictions = self.model.predict(X)
        if self.task_type == TaskType.CLASSIFICATION:
            predictions = np.argmax(predictions, axis=1)
        elif self.task_type == TaskType.REGRESSION and self.num_output == 1:
            predictions = predictions.flatten()
        return predictions