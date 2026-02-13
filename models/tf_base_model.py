#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base class and Wrapper for TensorFlow models - for integration with existing evaluation framework
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, Dict
import logging
import pickle
import os

logger = logging.getLogger(__name__)


class TFModelWrapper:
    """
    Wrapper class for TensorFlow models to conform to BaseModel interface
    """
    
    def __init__(self, model_class, name: str, dense_dim: int = 7, r_dim: int = 2, 
                 hidden_dim: int = 64, is_binary: int = 0, **model_kwargs):
        """
        Initialize TensorFlow model wrapper
        
        Args:
            model_class: TensorFlow model class
            name: Model name
            dense_dim: Non-monotonic feature dimension
            r_dim: Monotonic feature dimension (e.g., periods, cpm_thr)
            hidden_dim: Hidden layer dimension
            is_binary: Whether it's binary classification (0 for regression)
            **model_kwargs: Model-specific parameters
        """
        self.name = name
        self.model_class = model_class
        self.dense_dim = dense_dim
        self.r_dim = r_dim
        self.hidden_dim = hidden_dim
        self.is_binary = is_binary
        self.model_kwargs = model_kwargs
        self.is_trained = False
        
        # Create model instances
        self.pv_model = None
        self.uv_model = None
        self.optimizer_pv = None
        self.optimizer_uv = None
        
    def _split_features(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split features into non-monotonic features (x) and monotonic features (r)
        
        Assume the last r_dim features are monotonic features (periods, cpm_thr, etc.)
        """
        x = X[:, :-self.r_dim]  # Non-monotonic features
        r = X[:, -self.r_dim:]   # Monotonic features (periods, cpm_thr)
        return x, r
    
    def fit(self, X: np.ndarray, y_pv: np.ndarray, y_uv: np.ndarray, 
            theoretical_max: Optional[Dict] = None,
            epochs: int = 100, batch_size: int = 1024, learning_rate: float = 0.001) -> None:
        """
        Train TensorFlow model
        
        Args:
            X: Feature matrix
            y_pv: PV target values
            y_uv: UV target values
            theoretical_max: Theoretical maximum (not used yet)
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        logger.info(f"Training TensorFlow model {self.name} (epochs={epochs}, batch_size={batch_size})")
        
        # Update feature dimensions
        self.dense_dim = X.shape[1] - self.r_dim
        
        # Split features
        x, r = self._split_features(X)
        
        # Normalize PV and UV (TensorFlow models are sensitive to numerical ranges)
        self.pv_mean = np.mean(y_pv)
        self.pv_std = np.std(y_pv) + 1e-8
        self.uv_mean = np.mean(y_uv)
        self.uv_std = np.std(y_uv) + 1e-8
        
        y_pv_norm = (y_pv - self.pv_mean) / self.pv_std
        y_uv_norm = (y_uv - self.uv_mean) / self.uv_std
        
        # Create model instances
        self.pv_model = self.model_class(
            name=f"{self.name}_pv",
            dense_dim=self.dense_dim,
            r_dim=self.r_dim,
            hidden_dim=self.hidden_dim,
            is_binary=self.is_binary,
            **self.model_kwargs
        )
        
        self.uv_model = self.model_class(
            name=f"{self.name}_uv",
            dense_dim=self.dense_dim,
            r_dim=self.r_dim,
            hidden_dim=self.hidden_dim,
            is_binary=self.is_binary,
            **self.model_kwargs
        )
        
        # Create optimizers
        self.optimizer_pv = tf.optimizers.Adam(learning_rate=learning_rate)
        self.optimizer_uv = tf.optimizers.Adam(learning_rate=learning_rate)
        
        # Train PV model
        logger.info(f"Training PV model...")
        self._train_single_model(x, r, y_pv_norm, self.pv_model, self.optimizer_pv, 
                                epochs, batch_size, "PV")
        
        # Train UV model
        logger.info(f"Training UV model...")
        self._train_single_model(x, r, y_uv_norm, self.uv_model, self.optimizer_uv, 
                                epochs, batch_size, "UV")
        
        self.is_trained = True
        logger.info(f"TensorFlow model {self.name} training complete")
    
    def _train_single_model(self, x: np.ndarray, r: np.ndarray, y: np.ndarray,
                           model, optimizer, epochs: int, batch_size: int, 
                           target_name: str) -> None:
        """Train a single model (PV or UV)"""
        dataset = tf.data.Dataset.from_tensor_slices((
            (tf.constant(x, dtype=tf.float32), tf.constant(r, dtype=tf.float32)),
            tf.constant(y, dtype=tf.float32)
        ))
        dataset = dataset.shuffle(10000).batch(batch_size)
        
        for epoch in range(epochs):
            total_loss = 0
            batch_count = 0
            
            for batch_inputs, batch_y in dataset:
                with tf.GradientTape() as tape:
                    y_pred, loss = model(batch_inputs, batch_y)
                
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
                total_loss += loss.numpy()
                batch_count += 1
            
            avg_loss = total_loss / batch_count
            if (epoch + 1) % 1 == 0:
                logger.info(f"  Epoch {epoch+1}/{epochs}, {target_name} Loss: {avg_loss:.4f}")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict PV and UV
        
        Args:
            X: Feature matrix
            
        Returns:
            (y_pv_pred, y_uv_pred)
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Split features
        x, r = self._split_features(X)
        
        # Convert to TensorFlow tensors
        x_tf = tf.constant(x, dtype=tf.float32)
        r_tf = tf.constant(r, dtype=tf.float32)
        
        # Predict (use test mode to avoid randomness)
        y_pv_pred_norm, _ = self.pv_model((x_tf, r_tf), tf.zeros(x.shape[0]), is_test=True)
        y_uv_pred_norm, _ = self.uv_model((x_tf, r_tf), tf.zeros(x.shape[0]), is_test=True)
        
        # Denormalize
        y_pv_pred = y_pv_pred_norm.numpy() * self.pv_std + self.pv_mean
        y_uv_pred = y_uv_pred_norm.numpy() * self.uv_std + self.uv_mean
        
        # Apply constraint (non-negative)
        y_pv_pred = np.clip(y_pv_pred, 0, None)
        y_uv_pred = np.clip(y_uv_pred, 0, None)
        
        return y_pv_pred, y_uv_pred
    
    def save(self, path: str) -> None:
        """Save model"""
        if not self.is_trained:
            logger.warning(f"Model {self.name} not trained, save may be invalid")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model weights and metadata
        save_dict = {
            'name': self.name,
            'model_class_name': self.model_class.__name__,
            'dense_dim': self.dense_dim,
            'r_dim': self.r_dim,
            'hidden_dim': self.hidden_dim,
            'is_binary': self.is_binary,
            'model_kwargs': self.model_kwargs,
            'is_trained': self.is_trained,
            'pv_mean': self.pv_mean,
            'pv_std': self.pv_std,
            'uv_mean': self.uv_mean,
            'uv_std': self.uv_std,
            'pv_weights': [v.numpy() for v in self.pv_model.trainable_variables],
            'uv_weights': [v.numpy() for v in self.uv_model.trainable_variables],
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        logger.info(f"TensorFlow model {self.name} saved to: {path}")
    
    @classmethod
    def load(cls, path: str, model_class) -> 'TFModelWrapper':
        """Load model"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file does not exist: {path}")
        
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        # Create wrapper instance
        wrapper = cls(
            model_class=model_class,
            name=save_dict['name'],
            dense_dim=save_dict['dense_dim'],
            r_dim=save_dict['r_dim'],
            hidden_dim=save_dict['hidden_dim'],
            is_binary=save_dict['is_binary'],
            **save_dict['model_kwargs']
        )
        
        # Restore normalization parameters
        wrapper.pv_mean = save_dict['pv_mean']
        wrapper.pv_std = save_dict['pv_std']
        wrapper.uv_mean = save_dict['uv_mean']
        wrapper.uv_std = save_dict['uv_std']
        wrapper.is_trained = save_dict['is_trained']
        
        # Rebuild model and restore weights
        wrapper.pv_model = model_class(
            name=f"{wrapper.name}_pv",
            dense_dim=wrapper.dense_dim,
            r_dim=wrapper.r_dim,
            hidden_dim=wrapper.hidden_dim,
            is_binary=wrapper.is_binary,
            **wrapper.model_kwargs
        )
        
        wrapper.uv_model = model_class(
            name=f"{wrapper.name}_uv",
            dense_dim=wrapper.dense_dim,
            r_dim=wrapper.r_dim,
            hidden_dim=wrapper.hidden_dim,
            is_binary=wrapper.is_binary,
            **wrapper.model_kwargs
        )
        
        # Create optimizer (for restoration)
        wrapper.optimizer_pv = tf.optimizers.Adam()
        wrapper.optimizer_uv = tf.optimizers.Adam()
        
        # Restore weights (need to call model once first to initialize weights)
        dummy_x = tf.zeros((1, wrapper.dense_dim))
        dummy_r = tf.zeros((1, wrapper.r_dim))
        dummy_y = tf.zeros(1)
        wrapper.pv_model((dummy_x, dummy_r), dummy_y)
        wrapper.uv_model((dummy_x, dummy_r), dummy_y)
        
        # Set weights
        for var, weight in zip(wrapper.pv_model.trainable_variables, save_dict['pv_weights']):
            var.assign(weight)
        for var, weight in zip(wrapper.uv_model.trainable_variables, save_dict['uv_weights']):
            var.assign(weight)
        
        logger.info(f"TensorFlow model loaded from {path}")
        return wrapper
