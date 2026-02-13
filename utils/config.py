#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验配置管理工具
"""

import json
import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ExperimentConfig:
    """实验配置管理类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if self.config_path and os.path.exists(self.config_path):
            logger.info(f"从配置文件加载: {self.config_path}")
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        else:
            # 默认配置
            logger.info("使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "experiment_name": "RF_Contract_Inventory_Estimation",
            "data_config": {
                "data_path": "data/df_train1.pq",
                "test_size": 0.2,
                "random_state": 42,
                "split_types": ["random", "cpm", "schedule", "joint"]
            },
            "model_config": {
                "ridge": {"alpha": 1.0},
                "gbdt": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1
                },
                "mlp": {
                    "hidden_layer_sizes": [100, 50],
                    "activation": "relu",
                    "solver": "adam"
                },
                "tf_mlp": {
                    "hidden_dim": 64,
                    "epochs": 100,
                    "batch_size": 1024,
                    "learning_rate": 0.001
                },
                "tf_posnn": {
                    "hidden_dim": 64,
                    "epochs": 100,
                    "batch_size": 1024,
                    "learning_rate": 0.001
                }
            },
            "evaluation_config": {
                "metrics": ["mae", "rmse"],
                "monotonicity_check": True,
                "constraint_check": True
            },
            "output_config": {
                "output_dir": "experiments",
                "save_predictions": False,
                "save_models": True,
                "visualize": True,
                "n_vis_contexts": 3
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def get_data_config(self) -> Dict[str, Any]:
        """获取数据配置"""
        return self.config.get("data_config", {})
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """获取指定模型的配置"""
        return self.config.get("model_config", {}).get(model_name, {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """获取评估配置"""
        return self.config.get("evaluation_config", {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """获取输出配置"""
        return self.config.get("output_config", {})
    
    def save(self, path: str) -> None:
        """保存配置到文件"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)
        logger.info(f"配置已保存到: {path}")
    
    def update(self, updates: Dict[str, Any]) -> None:
        """更新配置"""
        self._deep_update(self.config, updates)
    
    def _deep_update(self, base: Dict, updates: Dict) -> None:
        """深度更新字典"""
        for key, value in updates.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
    
    def __str__(self) -> str:
        """字符串表示"""
        return json.dumps(self.config, indent=2, ensure_ascii=False)


def load_config(config_path: Optional[str] = None) -> ExperimentConfig:
    """
    加载实验配置
    
    Args:
        config_path: 配置文件路径，如果为None则尝试从默认位置加载
        
    Returns:
        ExperimentConfig实例
    """
    # 尝试从默认位置加载
    if config_path is None:
        default_paths = [
            "configs/experiment_config.json",
            "experiment_config.json",
        ]
        for path in default_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    return ExperimentConfig(config_path)


def create_config_template(output_path: str = "configs/my_experiment.json") -> None:
    """
    创建配置文件模板
    
    Args:
        output_path: 输出文件路径
    """
    config = ExperimentConfig()
    config.save(output_path)
    logger.info(f"配置模板已创建: {output_path}")
