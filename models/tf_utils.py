#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common utility functions for TensorFlow models
"""

import tensorflow as tf
import hashlib


def sg(x):
    """Stop gradient"""
    return tf.stop_gradient(x)


def get_variable(shape, name):
    """Initialize variable"""
    seed = tf.get_current_name_scope() + name
    seed = int(hashlib.md5(seed.encode()).hexdigest(), 16) & (2**32-1)
    if len(shape) == 2:
        l = 12. / (shape[0] + shape[1])
        res = tf.Variable(tf.random.uniform(shape, -l, l, seed=seed), name=name)
        return res
    else:
        return tf.Variable(tf.zeros(shape), name=name)


def get_embedding(shape, name):
    """Initialize embedding"""
    seed = tf.get_current_name_scope() + name
    seed = int(hashlib.md5(seed.encode()).hexdigest(), 16) & (2**32-1)
    res = tf.Variable(tf.random.normal(shape, stddev=0.001, seed=seed), name=name)
    return res


def pos_w(w, k=10.0):
    """Positive weight constraint"""
    return tf.nn.softplus(w * k) / k


def ce_loss(y, y_pred):
    """Cross-entropy loss"""
    return tf.reduce_mean(- y * tf.math.log(y_pred + 1e-5) - (1-y) * tf.math.log(1 - y_pred + 1e-5))


def mse_loss(y, y_pred):
    """Mean squared error loss"""
    return tf.reduce_mean(tf.square(y_pred-y))


def afunc(x):
    """Auxiliary activation function"""
    return (1 + tf.nn.relu(x)) / (1 + tf.nn.relu(-x)) + 0.1


class TFBaseModel(tf.Module):
    """TensorFlow model base class"""
    
    def __init__(self, dense_dim=33, r_dim=4, hidden_dim=16, is_binary=1, **kwargs):
        super().__init__(**kwargs)
        self.seed = int(hashlib.md5(self.name.encode()).hexdigest(), 16) & (2**32-1)
        self.dense_dim = dense_dim
        self.r_dim = r_dim
        self.hidden_dim = hidden_dim
        self.is_binary = is_binary
