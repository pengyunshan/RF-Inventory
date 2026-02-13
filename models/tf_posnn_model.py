#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
带正权重约束的神经网络模型 - 确保单调性
"""

import tensorflow as tf
from .tf_utils import TFBaseModel, get_variable, pos_w, ce_loss, mse_loss


class PosNNModel(TFBaseModel):
    """带正权重约束的神经网络，确保对单调特征的单调性"""
    
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
            # 对单调特征使用正权重约束
            s = x @ self.wx + r @ pos_w(self.wr) + self.b
            s = tf.nn.tanh(s)
            s = s @ pos_w(self.w1) + self.b1
            s = tf.nn.tanh(s)
            s = s @ pos_w(self.w2) + self.b2
            s = tf.squeeze(s, -1)
            if self.is_binary:
                y_pred = tf.nn.sigmoid(s)
                loss = ce_loss(y, y_pred)
            else:
                y_pred = s
                loss = mse_loss(y, y_pred)
            return y_pred, loss
