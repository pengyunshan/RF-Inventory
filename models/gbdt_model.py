#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Histogram-based Gradient Boosting Decision Tree model
Uses HistGradientBoostingRegressor, faster on medium to large datasets
"""

import numpy as np
from typing import Optional
from sklearn.ensemble import HistGradientBoostingRegressor
from .base_model import BaseModel
from typing import Tuple, Optional, Dict 

class GBDTModel(BaseModel):
    """
    Histogram-based Gradient Boosting Decision Tree model
    
    Uses HistGradientBoostingRegressor instead of GradientBoostingRegressor:
    - Faster on large datasets (n_samples >= 10000)
    - Supports monotonicity constraints
    - Native support for missing values
    - Binning strategy based on histograms
    """
    
    def __init__(self, max_iter: int = 100, max_depth: int = 6, 
                 learning_rate: float = 0.1, monotonic_cst: Optional[list] = None,
                 l2_regularization: float = 0.0, max_leaf_nodes: Optional[int] = None,
                 categorical_features: Optional[list] = None, min_samples_leaf: int = 20):
        """
        Initialize GBDT model
        
        Args:
            max_iter: Maximum iterations (equivalent to number of trees), recommended 100-300
            max_depth: Maximum tree depth, recommended 6-10 (can be deeper with fewer features)
            learning_rate: Learning rate, recommended 0.01-0.1
            monotonic_cst: List of monotonicity constraints, one value per feature:
                          1 for monotonically increasing, -1 for monotonically decreasing, 0 for no constraint
                          Example: [0, 0, 0, 0, 0, 1, 1] means last two features are monotonically increasing
            l2_regularization: L2 regularization strength, recommended 0.0-1.0
            max_leaf_nodes: Maximum leaf nodes, recommended 31-63 (None for unlimited)
            categorical_features: List of categorical feature indices, e.g., [0, 1, 2] means first 3 features are categorical
            min_samples_leaf: Minimum samples per leaf, recommended 20-50 (can be larger for large datasets)
        """
        super().__init__("GBDT")
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.monotonic_cst = monotonic_cst
        self.l2_regularization = l2_regularization
        self.max_leaf_nodes = max_leaf_nodes
        self.categorical_features = categorical_features
        self.min_samples_leaf = min_samples_leaf
        
        # Create PV model
        self.pv_model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            max_depth=max_depth,
            learning_rate=learning_rate,
            monotonic_cst=monotonic_cst,
            l2_regularization=l2_regularization,
            max_leaf_nodes=max_leaf_nodes,
            categorical_features=categorical_features,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            early_stopping='auto',  # Auto enable early stopping for large datasets
            verbose=0
        )
        
        # Create UV model
        self.uv_model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            max_depth=max_depth,
            learning_rate=learning_rate,
            monotonic_cst=monotonic_cst,
            l2_regularization=l2_regularization,
            max_leaf_nodes=max_leaf_nodes,
            categorical_features=categorical_features,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            early_stopping='auto',
            verbose=0
        )
    
    def fit(self, X: np.ndarray, y_pv: np.ndarray, y_uv: np.ndarray, 
            theoretical_max: Optional[Dict] = None) -> None:
        """Train GBDT model"""
        from utils.evaluation import logger
        logger.info(f"Training HistGBDT model (max_iter={self.max_iter}, max_depth={self.max_depth}, lr={self.learning_rate})")
        if self.monotonic_cst is not None:
            logger.info(f"  Monotonicity constraints: {self.monotonic_cst}")
        if self.categorical_features is not None:
            logger.info(f"  Categorical feature indices: {self.categorical_features}")
        if self.l2_regularization > 0:
            logger.info(f"  L2 regularization: {self.l2_regularization}")
        if self.max_leaf_nodes is not None:
            logger.info(f"  Maximum leaf nodes: {self.max_leaf_nodes}")
        logger.info(f"  Minimum samples per leaf: {self.min_samples_leaf}")
        
        self.pv_model.fit(X, y_pv)
        self.uv_model.fit(X, y_uv)
        self.is_trained = True
        
        # Output actual number of iterations (may be less than max_iter due to early stopping)
        logger.info(f"PV model actual iterations: {self.pv_model.n_iter_}")
        logger.info(f"UV model actual iterations: {self.uv_model.n_iter_}")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        y_pv_pred = self.pv_model.predict(X)
        y_uv_pred = self.uv_model.predict(X)
        
        # Apply non-negative constraint
        y_pv_pred = np.clip(y_pv_pred, 0, None)
        y_uv_pred = np.clip(y_uv_pred, 0, None)
        
        return y_pv_pred, y_uv_pred