#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提示学习模型 - 使用数据增强的提示学习方法
"""

import tensorflow as tf
from .tf_utils import TFBaseModel, get_variable, sg, ce_loss, mse_loss


class HintModel(TFBaseModel):
    """
    提示学习模型
    通过对单调特征进行微小扰动，学习单调性约束
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        dense_dim, r_dim, hidden_dim = self.dense_dim, self.r_dim, self.hidden_dim
        with tf.name_scope(f'{self.name}_net'):
            self.wx = get_variable([dense_dim, hidden_dim], 'wx')
            self.wr = get_variable([r_dim, hidden_dim], 'wr')
            self.b = get_variable([hidden_dim], 'b')
            self.w1 = get_variable([hidden_dim, hidden_dim], 'w1')
            self.b1 = get_variable([hidden_dim], 'b1')
            self.w2 = get_variable([hidden_dim, 1], 'w2')
            self.b2 = get_variable([1], 'b2')
    
    def __call__(self, inputs, y, is_test=False, **kwargs):
        x, r = inputs
        with tf.name_scope(f'{self.name}_net'):
            # 前向传播
            s = x @ self.wx + r @ self.wr + self.b
            s = tf.tanh(s)
            s = s @ self.w1 + self.b1
            s = tf.tanh(s)
            s = s @ self.w2 + self.b2
            s = tf.squeeze(s, -1)
            
            # 计算预测值和基础损失
            if self.is_binary:
                y_pred = tf.nn.sigmoid(s)
                loss = ce_loss(y, y_pred)
            else:
                y_pred = s
                loss = mse_loss(y, y_pred)
            
            # 如果不是测试模式，添加提示学习的正则化项
            if not is_test:
                # 生成随机扰动方向
                t = tf.sign(tf.random.uniform([r.shape[0], 1], seed=self.seed) - 0.5)
                self.seed += 1
                
                # 对单调特征添加小的随机扰动
                r_ = r + tf.random.uniform(r.shape, seed=self.seed) * 0.1 * t
                self.seed += 1
                
                # 计算扰动后的输出
                s_ = x @ self.wx + r_ @ self.wr + self.b
                s_ = tf.tanh(s_)
                s_ = s_ @ self.w1 + self.b1
                s_ = tf.tanh(s_)
                s_ = s_ @ self.w2 + self.b2
                s_ = tf.squeeze(s_, -1)
                
                # 提示损失：如果r增加，s应该增加（或至少不减少）
                # 使用stop_gradient避免梯度传播
                delta = tf.reduce_mean(tf.nn.relu((s - s_) * t) ** 2)
                delta -= sg(delta)  # 减去stop_gradient版本，只保留梯度
                loss += delta
            
            return y_pred, loss
