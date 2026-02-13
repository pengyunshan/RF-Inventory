#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Piecewise linear model - uses gradient regularization to ensure monotonicity
"""

import tensorflow as tf
from .tf_utils import TFBaseModel, get_variable, sg, ce_loss, mse_loss


class PwlModel(TFBaseModel):
    """
    Piecewise Linear (PWL) model
    Ensures output monotonicity with respect to monotonic features through gradient regularization
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
            # First layer
            h1 = x @ self.wx + r @ self.wr + self.b
            h1 = tf.tanh(h1)
            
            # Second layer
            h2 = h1 @ self.w1 + self.b1
            h2 = tf.tanh(h2)
            
            # Output layer
            s = h2 @ self.w2 + self.b2
            s = tf.squeeze(s, -1)
            
            # Calculate predictions and base loss
            if self.is_binary:
                y_pred = tf.nn.sigmoid(s)
                loss = ce_loss(y, y_pred)
            else:
                y_pred = s
                loss = mse_loss(y, y_pred)
            
            # Add gradient regularization to ensure monotonicity
            # Compute gradient of output with respect to monotonic features r
            # Gradient of h2: (1-tanh^2(h2)) @ W1^T
            h2_grads = (tf.transpose(self.w2) * (1 - tf.square(sg(h2)))) @ tf.transpose(self.w1)
            
            # Gradient of h1
            # Gradient of r: h2_grads * (1-tanh^2(h1)) @ Wr^T
            r_grads = (h2_grads * (1 - tf.square(sg(h1)))) @ tf.transpose(self.wr)
            
            # Regularization: penalize negative gradients (ensure positive gradient, i.e., monotonically increasing)
            reg = tf.reduce_sum(tf.nn.relu(-r_grads)) * 0.01
            reg -= sg(reg)  # Keep gradient only
            loss += reg
            
            return y_pred, loss
