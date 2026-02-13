#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Layer Perceptron模型
"""

import numpy as np
from typing import Optional, Tuple
from sklearn.neural_network import MLPRegressor
from .base_model import BaseModel
from typing import Tuple, Optional, Dict 

class MLPModel(BaseModel):
    """Multi-Layer Perceptron模型"""
    
    def __init__(self, hidden_layer_sizes: Tuple = (100, 50), 
                 activation: str = 'relu', solver: str = 'adam'):
        super().__init__("MLP")
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.pv_model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            random_state=42,
            max_iter=1000
        )
        self.uv_model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            random_state=42,
            max_iter=1000
        )
    
    def fit(self, X: np.ndarray, y_pv: np.ndarray, y_uv: np.ndarray, 
            theoretical_max: Optional[Dict] = None) -> None:
        """训练MLP模型"""
        from utils.evaluation import logger
        logger.info(f"训练MLP模型 (hidden_layers={self.hidden_layer_sizes})")
        
        self.pv_model.fit(X, y_pv)
        self.uv_model.fit(X, y_uv)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """预测"""
        if not self.is_trained:
            raise ValueError("模型未训练")
        
        y_pv_pred = self.pv_model.predict(X)
        y_uv_pred = self.uv_model.predict(X)
        
        # 应用非负约束
        y_pv_pred = np.clip(y_pv_pred, 0, None)
        y_uv_pred = np.clip(y_uv_pred, 0, None)
        
        return y_pv_pred, y_uv_pred