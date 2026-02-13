#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base model class
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict as DictType
import numpy as np
import logging
import pickle
import os

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Model base class"""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_trained = False
    
    @abstractmethod
    def fit(self, X: np.ndarray, y_pv: np.ndarray, y_uv: np.ndarray, 
            theoretical_max: Optional[DictType] = None) -> None:
        """Train model"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict impressions and unique visitors"""
        pass
    
    def evaluate(self, X_test: np.ndarray, y_pv_test: np.ndarray, 
                 y_uv_test: np.ndarray) -> dict:
        """Evaluate model performance"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        y_pv_pred, y_uv_pred = self.predict(X_test)
        
        # Calculate error metrics
        pv_mae = mean_absolute_error(y_pv_test, y_pv_pred)
        pv_rmse = np.sqrt(mean_squared_error(y_pv_test, y_pv_pred))
        uv_mae = mean_absolute_error(y_uv_test, y_uv_pred)
        uv_rmse = np.sqrt(mean_squared_error(y_uv_test, y_uv_pred))
        
        return {
            'pv_mae': pv_mae,
            'pv_rmse': pv_rmse,
            'uv_mae': uv_mae,
            'uv_rmse': uv_rmse
        }
    
    def save(self, path: str) -> None:
        """
        Save model to file
        
        Args:
            path: Save path (.pkl file)
        """
        if not self.is_trained:
            logger.warning(f"Model {self.name} not trained, save may be invalid")
        
        # Create directory
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        
        logger.info(f"Model {self.name} saved to: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'BaseModel':
        """
        Load model from file
        
        Args:
            path: Model file path
        
        Returns:
            Loaded model instance
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file does not exist: {path}")
        
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Model loaded from {path}")
        return model