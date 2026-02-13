#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model factory - manage creation and registration of all models
"""

import json
import os
from typing import List, Optional, Dict
from .base_model import BaseModel
from .ridge_model import RidgeModel
from .gbdt_model import GBDTModel
from .mlp_model import MLPModel
from .monotonic_mlp_model import MonotonicMLPModel
from .logistic_regression_model import LogisticRegressionModel, LogisticRegressionBaseline
from .nearest_fc_model import NearestFCRetrieval


def load_gbdt_config(config_name: str = "optimized") -> Dict:
    """
    Load GBDT parameters from config file
    
    Args:
        config_name: Configuration name, options:
            - "default": Default configuration (baseline)
            - "optimized": Optimized configuration (recommended)
            - "fast": Fast training configuration
            - "deep": Deep model configuration
            - "conservative": Conservative configuration (prevent overfitting)
            - "experimental": Experimental configuration
    
    Returns:
        GBDT parameter dictionary
    """
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'configs', 'gbdt_config.json'
    )
    
    if not os.path.exists(config_path):
        # If config file doesn't exist, return default optimized configuration
        return {
            'max_iter': 200,
            'max_depth': 8,
            'learning_rate': 0.05,
            'monotonic_cst': [0, 0, 0, 0, 0, 1, 1],
            'l2_regularization': 0.1,
            'max_leaf_nodes': 63,
            'categorical_features': [0, 1, 2],
            'min_samples_leaf': 30
        }
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    if config_name not in config['model_configs']:
        raise ValueError(
            f"Configuration '{config_name}' does not exist. Available: {list(config['model_configs'].keys())}"
        )
    
    return config['model_configs'][config_name]


def create_gbdt_from_config(config_name: str = "optimized") -> GBDTModel:
    """
    Create GBDT model from config file
    
    Args:
        config_name: Configuration name
    
    Returns:
        Configured GBDT model instance
    """
    params = load_gbdt_config(config_name)
    
    # Remove description field (not a model parameter)
    params_clean = {k: v for k, v in params.items() if k != 'description'}
    
    return GBDTModel(**params_clean)


def get_baseline_models(gbdt_config: str = "optimized") -> List[BaseModel]:
    """
    Get baseline models list
    
    Args:
        gbdt_config: GBDT configuration name, default "optimized"
    
    Returns:
        Model list
    """
    return [
        RidgeModel(alpha=1.0),
        create_gbdt_from_config(gbdt_config),
        MLPModel(hidden_layer_sizes=(100, 50)),
        MonotonicMLPModel(hidden_layer_sizes=(100, 50), 
                         monotonic_features=[-2, -1]),  # Last two features: periods, cpm_thr
        LogisticRegressionBaseline(alpha=0.0)
    ]


def get_all_models(gbdt_config: str = "optimized") -> List[BaseModel]:
    """
    Get all models list (including logistic regression)
    
    Args:
        gbdt_config: GBDT configuration name, default "optimized"
    
    Returns:
        Model list
    """
    return [
        RidgeModel(alpha=1.0),
        create_gbdt_from_config(gbdt_config),
        MLPModel(hidden_layer_sizes=(100, 50)),
        MonotonicMLPModel(hidden_layer_sizes=(100, 50), 
                         monotonic_features=[-2, -1]),  # Last two features: periods, cpm_thr
        LogisticRegressionModel(C=1.0),
        LogisticRegressionBaseline(alpha=0.0),
        NearestFCRetrieval()
    ]


def get_model_by_name(model_name: str, gbdt_config: str = "optimized") -> BaseModel:
    """
    Get model by name
    
    Args:
        model_name: Model name
        gbdt_config: GBDT configuration name (used only when model_name='gbdt')
    
    Returns:
        Model instance
    """
    model_map = {
        'ridge': lambda: RidgeModel(),
        'gbdt': lambda: create_gbdt_from_config(gbdt_config),
        'mlp': lambda: MLPModel(),
        'monotonic_mlp': lambda: MonotonicMLPModel(),
        'logistic_regression': lambda: LogisticRegressionModel(),
        'logistic_regression_baseline': lambda: LogisticRegressionBaseline(),
        'nearest_fc': lambda: NearestFCRetrieval()
    }
    
    if model_name.lower() in model_map:
        return model_map[model_name.lower()]()
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def get_model_names() -> List[str]:
    """Get all supported model names"""
    return ['ridge', 'gbdt', 'mlp', 'monotonic_mlp', 'logistic_regression', 
            'logistic_regression_baseline', 'nearest_fc']