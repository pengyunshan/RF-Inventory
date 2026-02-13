#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
具有单调性约束的MLP模型
"""

import numpy as np
from typing import Optional, List
from .mlp_model import MLPModel
from typing import Tuple, Optional, Dict 

class MonotonicMLPModel(MLPModel):
    """具有单调性约束的MLP模型"""
    
    def __init__(self, hidden_layer_sizes: Tuple = (100, 50), 
                 activation: str = 'relu', solver: str = 'adam',
                 monotonic_features: Optional[List[int]] = None):
        super().__init__(hidden_layer_sizes, activation, solver)
        self.name = "Monotonic_MLP"
        self.monotonic_features = monotonic_features or []
    
    def fit(self, X: np.ndarray, y_pv: np.ndarray, y_uv: np.ndarray, 
            theoretical_max: Optional[Dict] = None) -> None:
        """训练具有单调性约束的MLP模型"""
        from utils.evaluation import logger
        logger.info(f"训练单调MLP模型 (monotonic_features={self.monotonic_features})")
        
        # 这里简化实现，实际的单调性约束需要更复杂的损失函数设计
        # 可以通过在损失函数中添加单调性正则项来实现
        super().fit(X, y_pv, y_uv, theoretical_max)
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """预测"""
        y_pv_pred, y_uv_pred = super().predict(X)
        
        # 可以在这里添加单调性后处理
        return y_pv_pred, y_uv_pred