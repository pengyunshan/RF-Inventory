#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow模型工厂 - 用于创建和管理TensorFlow模型
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# 尝试导入TensorFlow
TF_AVAILABLE = False
try:
    import tensorflow as tf
    from .tf_base_model import TFModelWrapper
    from .tf_mlp_model import MlpModel as TFMlpModel
    from .tf_posnn_model import PosNNModel as TFPosNNModel
    from .tf_minmax_model import MinmaxModel as TFMinmaxModel
    from .tf_smoothed_minmax_model import SmoothedMinmaxModel as TFSmoothedMinmaxModel
    from .tf_constrained_model import ConstrainedModel as TFConstrainedModel
    from .tf_scalable_model import ScalableModel as TFScalableModel
    from .tf_hint_model import HintModel as TFHintModel
    from .tf_pwl_model import PwlModel as TFPwlModel
    from .tf_gcm_model import GcmModel as TFGcmModel
    from .tf_igcm_model import IgcmModel as TFIgcmModel
    TF_AVAILABLE = True
    logger.info("TensorFlow可用，已加载TensorFlow模型")
except ImportError as e:
    logger.warning(f"TensorFlow不可用: {e}，TensorFlow模型将被禁用")
    logger.warning("如需使用TensorFlow模型，请安装: pip install tensorflow>=2.10.0 tensorflow-probability>=0.18.0")


def get_tf_models(hidden_dim: int = 64, r_dim: int = 2, 
                  epochs: int = 100) -> List:
    """
    获取所有TensorFlow模型
    
    Args:
        hidden_dim: 隐藏层维度
        r_dim: 单调特征维度（periods, cpm_thr）
        epochs: 训练轮数
        
    Returns:
        TensorFlow模型列表
    """
    if not TF_AVAILABLE:
        logger.warning("TensorFlow不可用，返回空列表")
        return []
    
    models = []
    
    # 基础模型
    for model_class, name in [
        (TFMlpModel, "TF_MLP"),
        (TFPosNNModel, "TF_PosNN"),
        (TFMinmaxModel, "TF_MinMax"),
        (TFSmoothedMinmaxModel, "TF_SmoothedMinMax"),
        (TFConstrainedModel, "TF_Constrained"),
        (TFScalableModel, "TF_Scalable"),
        (TFHintModel, "TF_Hint"),
        (TFPwlModel, "TF_Pwl"),
    ]:
        wrapper = TFModelWrapper(
            model_class=model_class,
            name=name,
            r_dim=r_dim,
            hidden_dim=hidden_dim,
            is_binary=0
        )
        wrapper.epochs = epochs
        models.append(wrapper)
    
    # 生成模型（需要tensorflow-probability）
    try:
        for model_class, name, kwargs in [
            (TFGcmModel, "TF_GCM", {}),
            (TFIgcmModel, "TF_IGCM", {}),
        ]:
            wrapper = TFModelWrapper(
                model_class=model_class,
                name=name,
                r_dim=r_dim,
                hidden_dim=hidden_dim,
                is_binary=0,
                **kwargs
            )
            wrapper.epochs = epochs
            models.append(wrapper)
    except ImportError:
        logger.warning("tensorflow-probability不可用，GCM和IGCM模型将被禁用")
    
    return models


def get_tf_model_by_name(model_name: str, hidden_dim: int = 64, 
                         r_dim: int = 2, epochs: int = 100) -> Optional[object]:
    """
    根据名称获取TensorFlow模型
    
    Args:
        model_name: 模型名称（不区分大小写）
        hidden_dim: 隐藏层维度
        r_dim: 单调特征维度
        epochs: 训练轮数
        
    Returns:
        模型实例或None
    """
    if not TF_AVAILABLE:
        return None
    
    model_name_lower = model_name.lower().replace('_', '').replace('-', '')
    
    # 模型映射表
    model_map = {
        'tfmlp': (TFMlpModel, "TF_MLP", {}),
        'tensorflowmlp': (TFMlpModel, "TF_MLP", {}),
        'tfposnn': (TFPosNNModel, "TF_PosNN", {}),
        'posnn': (TFPosNNModel, "TF_PosNN", {}),
        'tfminmax': (TFMinmaxModel, "TF_MinMax", {}),
        'minmax': (TFMinmaxModel, "TF_MinMax", {}),
        'tfsmoothedminmax': (TFSmoothedMinmaxModel, "TF_SmoothedMinMax", {}),
        'smoothedminmax': (TFSmoothedMinmaxModel, "TF_SmoothedMinMax", {}),
        'tfconstrained': (TFConstrainedModel, "TF_Constrained", {}),
        'constrained': (TFConstrainedModel, "TF_Constrained", {}),
        'tfscalable': (TFScalableModel, "TF_Scalable", {}),
        'scalable': (TFScalableModel, "TF_Scalable", {}),
        'tfhint': (TFHintModel, "TF_Hint", {}),
        'hint': (TFHintModel, "TF_Hint", {}),
        'tfpwl': (TFPwlModel, "TF_Pwl", {}),
        'pwl': (TFPwlModel, "TF_Pwl", {}),
        'tfgcm': (TFGcmModel, "TF_GCM", {}),
        'gcm': (TFGcmModel, "TF_GCM", {}),
        'tfigcm': (TFIgcmModel, "TF_IGCM", {}),
        'igcm': (TFIgcmModel, "TF_IGCM", {}),
    }
    
    if model_name_lower in model_map:
        model_class, name, kwargs = model_map[model_name_lower]
        try:
            wrapper = TFModelWrapper(
                model_class=model_class,
                name=name,
                r_dim=r_dim,
                hidden_dim=hidden_dim,
                is_binary=0,
                **kwargs
            )
            wrapper.epochs = epochs
            return wrapper
        except ImportError as e:
            logger.error(f"创建模型 {name} 失败: {e}")
            return None
    
    return None


def is_tf_model(model) -> bool:
    """判断是否是TensorFlow模型"""
    if not TF_AVAILABLE:
        return False
    return isinstance(model, TFModelWrapper)


def get_tf_model_names() -> List[str]:
    """获取所有TensorFlow模型名称"""
    if not TF_AVAILABLE:
        return []
    return [
        "tf_mlp", "tf_posnn", "tf_minmax", "tf_smoothed_minmax",
        "tf_constrained", "tf_scalable", "tf_hint", "tf_pwl",
        "tf_gcm", "tf_igcm"
    ]

