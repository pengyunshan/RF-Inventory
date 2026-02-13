#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MinMax model - uses Min-Max operations to ensure monotonicity
"""

import tensorflow as tf
from .tf_utils import TFBaseModel, get_variable, pos_w, ce_loss, mse_loss


class MinmaxModel(TFBaseModel):
    """
    MinMax model
    Builds a network with specific structure using min and max operations to ensure monotonicity
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        dense_dim, r_dim, hidden_dim = self.dense_dim, self.r_dim, self.hidden_dim
        with tf.name_scope(f'{self.name}_net'):
            # Non-monotonic feature processing branch
            self.wx = get_variable([dense_dim, hidden_dim], 'wx')
            self.bx = get_variable([hidden_dim], 'bx')
            self.wx1 = get_variable([hidden_dim, hidden_dim], 'wx1')
            self.bx1 = get_variable([hidden_dim], 'bx1')
            self.wx2 = get_variable([hidden_dim, 9], 'wx2')
            self.b2 = get_variable([9], 'b2')
            
            # Monotonic feature processing branch (with positive weight constraint)
            self.wr = get_variable([r_dim, hidden_dim], 'wr')
            self.br = get_variable([hidden_dim], 'br')
            self.wr1 = get_variable([hidden_dim, hidden_dim], 'wr1')
            self.br1 = get_variable([hidden_dim], 'br1')
            self.wr2 = get_variable([hidden_dim, 9], 'wr2')
    
    def __call__(self, inputs, y, is_test=False, **kwargs):
        x, r = inputs
        with tf.name_scope(f'{self.name}_net'):
            # Process non-monotonic features
            x = x @ self.wx + self.bx
            xr = r @ pos_w(self.wr) + self.br + x  # Fuse monotonic features
            x = tf.nn.tanh(x)
            xr = tf.nn.tanh(xr)
            
            # Second layer
            x = x @ self.wx1 + self.bx1
            xr = xr @ pos_w(self.wr1) + self.br1 + x
            x = tf.nn.tanh(x)
            xr = tf.nn.tanh(xr)
            
            # Output layer: combine into 3x3 matrix, using min-max operations
            s = xr @ pos_w(self.wr2) + x @ self.wx2 + self.b2
            s = tf.reshape(s, [-1, 3, 3])  # Reshape to [batch, 3, 3]
            s = tf.reduce_min(s, -1)       # Take min on last dimension -> [batch, 3]
            s = tf.reduce_max(s, -1)       # Take max on last dimension -> [batch]
            
            if self.is_binary:
                y_pred = tf.nn.sigmoid(s)
                loss = ce_loss(y, y_pred)
            else:
                y_pred = s
                loss = mse_loss(y, y_pred)
            
            return y_pred, loss
