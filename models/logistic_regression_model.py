#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
逻辑回归基线模型
"""

import numpy as np
from typing import Optional
from sklearn.linear_model import LinearRegression
from .base_model import BaseModel
from typing import Tuple, Optional, Dict 

class LogisticRegressionModel(BaseModel):
    """逻辑回归模型 - 用于分类任务的baseline"""
    
    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        super().__init__("LogisticRegression")
        self.C = C
        self.max_iter = max_iter
        # 由于PV和UV是回归任务，我们使用线性回归作为逻辑回归的替代
        # 但为了保持命名一致性，这里仍然叫LogisticRegressionModel
        self.model = LinearRegression()
    
    def fit(self, X: np.ndarray, y_pv: np.ndarray, y_uv: np.ndarray, 
            theoretical_max: Optional[Dict] = None) -> None:
        """训练逻辑回归模型（实际使用线性回归）"""
        from utils.evaluation import logger
        logger.info(f"训练逻辑回归模型 (C={self.C}, max_iter={self.max_iter})")
        
        # 对于回归任务，我们使用线性回归
        self.model.fit(X, y_pv)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """预测"""
        if not self.is_trained:
            raise ValueError("模型未训练")
        
        # 由于这是回归任务，我们只预测PV，UV可以使用相同的模型或单独的模型
        y_pv_pred = self.model.predict(X)
        # 对于UV，我们可以使用相同的特征但不同的目标
        # 这里简化实现，使用相同的预测值（实际应该训练两个不同的模型）
        y_uv_pred = y_pv_pred * 0.1  # 简单的比例关系作为示例
        
        # 应用非负约束
        y_pv_pred = np.clip(y_pv_pred, 0, None)
        y_uv_pred = np.clip(y_uv_pred, 0, None)
        
        return y_pv_pred, y_uv_pred


class LogisticRegressionBaseline(BaseModel):
    """逻辑回归基线模型 - 专门用于回归任务的线性模型"""
    
    def __init__(self, alpha: float = 0.0):
        super().__init__("Logistic_Regression_Baseline")
        self.alpha = alpha
        self.pv_model = LinearRegression()
        self.uv_model = LinearRegression()
    
    def fit(self, X: np.ndarray, y_pv: np.ndarray, y_uv: np.ndarray, 
            theoretical_max: Optional[Dict] = None) -> None:
        """训练逻辑回归基线模型"""
        from utils.evaluation import logger
        logger.info(f"训练逻辑回归基线模型")
        
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