#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成约束模型 - 使用变分推断的生成模型
"""

import tensorflow as tf
try:
    import tensorflow_probability as tfp
    TFP_AVAILABLE = True
except ImportError:
    TFP_AVAILABLE = False
    
from .tf_utils import TFBaseModel, get_variable, pos_w, sg, afunc


class GcmModel(TFBaseModel):
    """
    生成约束模型（Generative Constraint Model）
    使用变分自编码器框架建模约束，通过隐变量z建模不确定性
    
    注意：需要安装tensorflow-probability
    """
    
    def __init__(self, sample_num=32, test_sample_num=128, z_dim=4, 
                 loss_type=0, beta=1, **kwargs):
        super().__init__(**kwargs)
        if not TFP_AVAILABLE:
            raise ImportError("GcmModel需要tensorflow-probability，请安装: pip install tensorflow-probability")
        
        dense_dim, r_dim, hidden_dim = self.dense_dim, self.r_dim, self.hidden_dim
        with tf.name_scope(f'{self.name}_net'):
            self.loss_type = loss_type
            self.sample_num = sample_num
            self.test_sample_num = test_sample_num
            self.beta = beta
            
            # 测试时使用的固定隐变量
            self.test_z = tf.constant(tf.random.normal([test_sample_num, z_dim], seed=self.seed))
            self.seed += 1
            
            # 编码器：x -> z
            self.wx = get_variable([dense_dim, hidden_dim], 'wx')
            self.b = get_variable([hidden_dim], 'b')
            self.wmu = get_variable([hidden_dim, z_dim], 'wmu')
            self.bmu = get_variable([z_dim], 'bmu')
            self.wsig = get_variable([hidden_dim, z_dim], 'wsig')
            self.bsig = get_variable([z_dim], 'bsig')
            
            # 解码器：z -> r的分布参数
            self.wh = get_variable([z_dim, hidden_dim], 'wh')
            self.bh = get_variable([hidden_dim], 'bh')
            self.wmc = get_variable([hidden_dim, r_dim], 'wmc')
            self.bmc = get_variable([r_dim], 'bmc')
            self.wsc = get_variable([hidden_dim, r_dim], 'wsc')
            self.bsc = get_variable([r_dim], 'bsc')
            
            # r的变换
            self.wr = get_variable([r_dim, r_dim], 'wr')
            
            # 如果loss_type==2，需要额外的网络
            if loss_type == 2:
                self.wt1 = get_variable([z_dim, hidden_dim], 'wt1')
                self.bt1 = get_variable([hidden_dim], 'bt1')
                self.wt2 = get_variable([hidden_dim, 1], 'wt2')
                self.bt2 = get_variable([1], 'bt2')
                self.wsy1 = get_variable([z_dim, hidden_dim], 'wsy1')
                self.bsy1 = get_variable([hidden_dim], 'bsy1')
                self.wsy2 = get_variable([hidden_dim, 1], 'wsy2')
                self.bsy2 = get_variable([1], 'bsy2')
    
    def __call__(self, inputs, y, is_test=False, **kwargs):
        x, r = inputs
        sample_num = self.sample_num if not is_test else self.test_sample_num
        
        with tf.name_scope(f'{self.name}_net'):
            # 编码器：从x编码到z的分布
            h = x @ self.wx + self.b
            h = tf.nn.tanh(h)
            mu = h @ self.wmu + self.bmu  # [batch, z_dim]
            log_var = h @ self.wsig + self.bsig  # [batch, z_dim]
            
            # 重参数化采样
            mu = tf.tile(tf.expand_dims(mu, 1), [1, sample_num, 1])  # [batch, k, z_dim]
            sig = tf.tile(tf.expand_dims(tf.math.exp(0.5 * log_var), 1), [1, sample_num, 1])
            
            rnd = self.test_z if is_test else tf.random.normal(tf.shape(mu), seed=self.seed)
            self.seed += 1
            z = rnd * sig + mu  # [batch, k, z_dim]
            
            # 计算概率
            log_p_z = tf.reduce_sum(tfp.distributions.Normal(loc=0, scale=1).log_prob(z), -1)  # [batch, k]
            log_q_z = tf.reduce_sum(tfp.distributions.Normal(loc=mu, scale=sig).log_prob(z), -1)  # [batch, k]
            
            # 解码器：从z解码到r的约束
            h = z @ self.wh + self.bh  # [batch, k, hidden_dim]
            h = tf.nn.tanh(h)
            mu_c = h @ self.wmc + self.bmc  # [batch, k, r_dim]
            s_c = afunc(h @ self.wsc + self.bsc)  # [batch, k, r_dim]
            
            # 计算约束满足概率
            s = (tf.expand_dims(r @ pos_w(self.wr), 1) - mu_c) / s_c  # [batch, k, r_dim]
            prob = tf.nn.sigmoid(s)
            prob = tf.reduce_prod(prob, -1)  # [batch, k]
            
            _y = tf.expand_dims(y, -1)
            
            if self.loss_type == 0:
                # 标准ELBO
                y_pred = tf.reduce_mean(prob, 1)  # [batch]
                log_p_y = _y * tf.math.log(prob + 1e-5) + (1 - _y) * tf.math.log(1 - prob + 1e-5)
                elb = tf.reduce_logsumexp(log_p_y + self.beta * sg(log_p_z - log_q_z), -1) - tf.math.log(float(sample_num))
                elb = tf.reduce_mean(elb)
                loss = -elb
                
            elif self.loss_type == 1:
                # 带偏移的ELBO
                tmp = tf.reduce_mean(prob, 1)
                y_pred = (tmp - 0.1) / 0.8
                log_p_y = (_y * 0.8 + 0.1) * tf.math.log(prob + 1e-5) + (1 - (_y * 0.8 + 0.1)) * tf.math.log(1 - prob + 1e-5)
                elb = tf.reduce_logsumexp(log_p_y + self.beta * sg(log_p_z - log_q_z), -1) - tf.math.log(float(sample_num))
                elb = tf.reduce_mean(elb)
                loss = -elb
                
            else:
                # loss_type == 2: 更复杂的建模
                t = z @ self.wt1 + self.bt1
                t = tf.nn.tanh(t)
                t = t @ self.wt2 + self.bt2
                t = tf.squeeze(t, -1)  # [batch, k]
                
                s_y = z @ self.wsy1 + self.bsy1
                s_y = tf.nn.tanh(s_y)
                s_y = s_y @ self.wsy2 + self.bsy2
                s_y = afunc(s_y)
                s_y = tf.squeeze(s_y, -1)  # [batch, k]
                
                mu_y = t + tf.math.log((1e-5 + prob) / (1e-5 + 1 - prob)) * s_y
                y_pred = tf.reduce_mean(mu_y, -1)
                
                tmp = tf.nn.sigmoid((tf.expand_dims(y, -1) - mu_y) / s_y)
                log_p_y = tf.math.log((1e-5 + tmp) * (1e-5 + 1 - tmp) / s_y)
                elb = tf.reduce_logsumexp(log_p_y + self.beta * sg(log_p_z - log_q_z), -1) - tf.math.log(float(sample_num))
                elb = tf.reduce_mean(elb)
                loss = -elb
            
            return y_pred, loss
