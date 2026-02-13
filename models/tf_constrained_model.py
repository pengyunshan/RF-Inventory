#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
约束模型 - 使用特殊激活函数确保单调性约束
"""

import tensorflow as tf
from .tf_utils import TFBaseModel, get_variable, pos_w, ce_loss, mse_loss


class ConstrainedModel(TFBaseModel):
    """
    约束模型
    使用特殊设计的激活函数确保对单调特征的单调性约束
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        dense_dim, r_dim, hidden_dim = self.dense_dim, self.r_dim, self.hidden_dim
        with tf.name_scope(f'{self.name}_net'):
            # 非单调特征处理分支
            self.wx = get_variable([dense_dim, hidden_dim], 'wx')
            self.bx = get_variable([hidden_dim], 'bx')
            self.wx1 = get_variable([hidden_dim, hidden_dim], 'wx1')
            self.bx1 = get_variable([hidden_dim], 'bx1')
            self.wx2 = get_variable([hidden_dim, 1], 'wx2')
            self.b2 = get_variable([1], 'b2')
            
            # 单调特征处理分支
            self.wr = get_variable([r_dim, hidden_dim], 'wr')
            self.br = get_variable([hidden_dim], 'br')
            self.wr1 = get_variable([hidden_dim, hidden_dim], 'wr1')
            self.br1 = get_variable([hidden_dim], 'br1')
            self.wr2 = get_variable([hidden_dim, 1], 'wr2')
    
    def __call__(self, inputs, y, is_test=False, **kwargs):
        x, r = inputs
        with tf.name_scope(f'{self.name}_net'):
            # 定义特殊激活函数，确保单调性
            f1 = lambda x: tf.nn.elu(x)           # 递增
            f2 = lambda x: -tf.nn.elu(-x)          # 递增
            f3 = lambda x: f2(tf.nn.relu(x)-1) + f1(-tf.nn.relu(-x)+1)  # 组合
            
            # 将隐藏层分成3部分，分别应用不同的激活函数
            t = int(self.hidden_dim / 3)
            fr = lambda x: tf.concat([
                f1(x[:, :t]), 
                f2(x[:, t:2*t]), 
                f3(x[:, 2*t:])
            ], -1)
            
            # 第一层
            x = x @ self.wx + self.bx
            xr = r @ pos_w(self.wr) + self.br + x
            x = tf.nn.tanh(x)
            xr = fr(xr)  # 应用特殊激活函数
            
            # 第二层
            x = x @ self.wx1 + self.bx1
            xr = xr @ pos_w(self.wr1) + self.br1 + x
            x = tf.nn.tanh(x)
            xr = fr(xr)
            
            # 输出层
            s = x @ self.wx2 + xr @ pos_w(self.wr2) + self.b2
            s = tf.squeeze(s, -1)
            
            if self.is_binary:
                y_pred = tf.nn.sigmoid(s)
                loss = ce_loss(y, y_pred)
            else:
                y_pred = s
                loss = mse_loss(y, y_pred)
            
            return y_pred, loss
