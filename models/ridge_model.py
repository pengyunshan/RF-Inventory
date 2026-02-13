#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ridge回归模型
"""

import numpy as np
from typing import Optional
from sklearn.linear_model import Ridge
from .base_model import BaseModel
from typing import Tuple, Optional, Dict 
class RidgeModel(BaseModel):
    """Ridge回归模型"""
    
    def __init__(self, alpha: float = 1.0):
        super().__init__("Ridge")
        self.alpha = alpha
        self.pv_model = Ridge(alpha=alpha)
        self.uv_model = Ridge(alpha=alpha)
    
    def fit(self, X: np.ndarray, y_pv: np.ndarray, y_uv: np.ndarray, 
            theoretical_max: Optional[Dict] = None) -> None:
        """训练Ridge模型"""
        from utils.evaluation import logger
        logger.info(f"训练Ridge模型 (alpha={self.alpha})")
        
        self.pv_model.fit(X, y_pv)
        self.uv_model.fit(X, y_uv)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """预测"""
        if not self.is_trained:
            raise ValueError("模型未训练")
        
        y_pv_pred = self.pv_model.predict(X)
        y_uv_pred = self.uv_model.predict(X)
        
        # 应用理论最大曝光量约束
        y_pv_pred = np.clip(y_pv_pred, 0, None)
        y_uv_pred = np.clip(y_uv_pred, 0, None)
        
        return y_pv_pred, y_uv_pred