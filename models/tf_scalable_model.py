#!/usr:bin/env python3
# -*- coding: utf-8 -*-
"""
可扩展模型 - 使用多分支网络结构
"""

import tensorflow as tf
from .tf_utils import TFBaseModel, get_variable, pos_w, ce_loss, mse_loss


class ScalableModel(TFBaseModel):
    """
    可扩展模型
    使用多分支网络结构，提高模型表达能力
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        dense_dim, r_dim, hidden_dim = self.dense_dim, self.r_dim, self.hidden_dim
        with tf.name_scope(f'{self.name}_net'):
            # 分支1
            self.wx10 = get_variable([dense_dim, hidden_dim], 'wx10')
            self.bx10 = get_variable([hidden_dim], 'bx10')
            self.wx11 = get_variable([hidden_dim, hidden_dim], 'wx11')
            self.bx11 = get_variable([hidden_dim], 'bx11')
            self.wx12 = get_variable([hidden_dim, 1], 'wx12')
            self.wx121 = get_variable([hidden_dim, hidden_dim], 'wx121')
            self.bx121 = get_variable([hidden_dim], 'bx121')
            
            # 分支2
            self.wx20 = get_variable([dense_dim, hidden_dim], 'wx20')
            self.bx20 = get_variable([hidden_dim], 'bx20')
            self.wx21 = get_variable([hidden_dim, hidden_dim], 'wx21')
            self.wx22 = get_variable([hidden_dim, 1], 'wx22')
            self.b2 = get_variable([1], 'b2')
            
            # 单调特征分支
            self.wr = get_variable([r_dim, hidden_dim], 'wr')
            self.br = get_variable([hidden_dim], 'br')
            self.wr1 = get_variable([hidden_dim, hidden_dim], 'wr1')
            self.br1 = get_variable([hidden_dim], 'br1')
            self.wr2 = get_variable([hidden_dim, 1], 'wr2')
    
    def __call__(self, inputs, y, is_test=False, **kwargs):
        x, r = inputs
        with tf.name_scope(f'{self.name}_net'):
            # 定义有界ReLU激活函数
            relun = lambda x: tf.nn.relu(x) - tf.nn.relu(x - 2)
            
            # 单调特征分支
            xr = r @ pos_w(self.wr) + self.br
            
            # 非单调特征分支1和分支2
            x1 = x @ self.wx10 + self.bx10
            x2 = x @ self.wx20 + self.bx20
            
            # 第一层激活
            xr = relun(xr)
            x1 = tf.nn.relu(x1)
            x2 = relun(x2)
            
            # 第二层
            xr = xr @ pos_w(self.wr1) + self.br1 + x2 @ self.wx21
            x2 = x1 @ self.wx121 + self.bx121
            x1 = x1 @ self.wx11 + self.bx11
            
            # 第二层激活
            xr = relun(xr)
            x1 = tf.nn.relu(x1)
            x2 = relun(x2)
            
            # 输出层：融合三个分支
            s = x1 @ self.wx12 + x2 @ self.wx22 + xr @ pos_w(self.wr2) + self.b2
            s = tf.squeeze(s, -1)
            
            if self.is_binary:
                y_pred = tf.nn.sigmoid(s)
                loss = ce_loss(y, y_pred)
            else:
                y_pred = s
                loss = mse_loss(y, y_pred)
            
            return y_pred, loss
