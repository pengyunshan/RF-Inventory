#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的生成约束模型 - 使用核技巧的变分推断
"""

import tensorflow as tf
try:
    import tensorflow_probability as tfp
    TFP_AVAILABLE = True
except ImportError:
    TFP_AVAILABLE = False
    
from .tf_utils import TFBaseModel, get_variable, pos_w, sg, afunc


class IgcmModel(TFBaseModel):
    """
    改进的生成约束模型（Improved Generative Constraint Model）
    在GCM基础上引入核（kernel）机制，更好地建模r的分布
    
    注意：需要安装tensorflow-probability
    """
    
    def __init__(self, sample_num=32, test_sample_num=128, z_dim=4, 
                 kern_dim=None, loss_type=0, beta=1, **kwargs):
        super().__init__(**kwargs)
        if not TFP_AVAILABLE:
            raise ImportError("IgcmModel需要tensorflow-probability，请安装: pip install tensorflow-probability")
        
        dense_dim, r_dim, hidden_dim = self.dense_dim, self.r_dim, self.hidden_dim
        if kern_dim is None:
            kern_dim = r_dim
        
        with tf.name_scope(f'{self.name}_net'):
            self.loss_type = loss_type
            self.sample_num = sample_num
            self.test_sample_num = test_sample_num
            self.beta = beta
            self.kern_dim = kern_dim
            
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
            
            # 解码器：z -> 约束
            self.wh = get_variable([z_dim, hidden_dim], 'wh')
            self.bh = get_variable([hidden_dim], 'bh')
            self.wmc = get_variable([hidden_dim, kern_dim], 'wmc')
            self.bmc = get_variable([kern_dim], 'bmc')
            self.wsc = get_variable([hidden_dim, kern_dim], 'wsc')
            self.bsc = get_variable([kern_dim], 'bsc')
            
            # r -> kernel编码器
            self.wr2kern = get_variable([r_dim, hidden_dim], 'wr2kern')
            self.br2kern = get_variable([hidden_dim], 'br2kern')
            self.wr2kernmu = get_variable([hidden_dim, kern_dim], 'wr2kernmu')
            self.br2kernmu = get_variable([kern_dim], 'br2kernmu')
            self.wr2kernsig = get_variable([hidden_dim, kern_dim], 'wr2kernsig')
            self.br2kernsig = get_variable([kern_dim], 'br2kernsig')
            
            # kernel -> r解码器
            self.wkern2r = get_variable([kern_dim, r_dim], 'wkern2r')
            self.bkern2r = get_variable([r_dim], 'bkern2r')
            self.rsig = get_variable([r_dim], 'rsig')
            
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
            # 编码器：x -> z的分布
            h = x @ self.wx + self.b
            h = tf.nn.tanh(h)
            mu = h @ self.wmu + self.bmu
            log_var = h @ self.wsig + self.bsig
            
            # 重参数化采样
            mu = tf.tile(tf.expand_dims(mu, 1), [1, sample_num, 1])
            sig = tf.tile(tf.expand_dims(tf.math.exp(0.5 * log_var), 1), [1, sample_num, 1])
            
            rnd = self.test_z if is_test else tf.random.normal(tf.shape(mu), seed=self.seed)
            self.seed += 1
            z = rnd * sig + mu  # [batch, k, z_dim]
            
            # 计算概率
            log_p_z = tf.reduce_sum(tfp.distributions.Normal(loc=0, scale=1).log_prob(z), -1)
            log_q_z = tf.reduce_sum(tfp.distributions.Normal(loc=mu, scale=sig).log_prob(z), -1)
            
            # 解码器：z -> 约束参数
            h = z @ self.wh + self.bh
            h = tf.nn.tanh(h)
            mu_c = h @ self.wmc + self.bmc
            s_c = afunc(h @ self.wsc + self.bsc)
            
            # 核编码器：r -> kernel
            h_kern = r @ self.wr2kern + self.br2kern
            h_kern = tf.nn.tanh(h_kern)
            mu_kern = h_kern @ self.wr2kernmu + self.br2kernmu
            logvar_kern = h_kern @ self.wr2kernsig + self.br2kernsig
            sig_kern = tf.math.exp(0.5 * logvar_kern)
            
            # 采样kernel
            kern = tf.random.normal(tf.shape(mu_kern), seed=self.seed)
            self.seed += 1
            kern = mu_kern + kern * sig_kern
            
            # 核解码器：kernel -> r的重建
            r_hat_mu = kern @ pos_w(self.wkern2r) + self.bkern2r
            r_hat_logsig = self.rsig
            r_hat_sig = tf.math.exp(r_hat_logsig)
            
            # r的ELBO
            r_elb = tf.reduce_sum(
                - 0.5 * tf.square((r - r_hat_mu) / r_hat_sig) - r_hat_logsig, -1
            ) + 0.5 * tf.reduce_sum(
                1 + logvar_kern - mu_kern ** 2 - sig_kern ** 2, -1
            )
            r_elb = tf.reduce_mean(r_elb) * 0.01
            r_elb -= sg(r_elb)  # 只保留梯度
            
            # 计算约束满足概率
            s = (tf.expand_dims(mu_kern, 1) - mu_c) / s_c  # [batch, k, kern_dim]
            prob = tf.nn.sigmoid(s)
            prob = tf.reduce_prod(prob, -1)  # [batch, k]
            
            _y = tf.expand_dims(y, -1)
            
            if self.loss_type == 0:
                # 标准ELBO
                y_pred = tf.reduce_mean(prob, 1)
                log_p_y = _y * tf.math.log(prob + 1e-5) + (1 - _y) * tf.math.log(1 - prob + 1e-5)
                elb = tf.reduce_logsumexp(
                    log_p_y + self.beta * sg(log_p_z - log_q_z), -1
                ) - tf.math.log(float(sample_num))
                elb = tf.reduce_mean(elb)
                loss = -elb - r_elb
                
            elif self.loss_type == 1:
                # 带偏移的ELBO
                tmp = tf.reduce_mean(prob, 1)
                y_pred = (tmp - 0.1) / 0.8
                log_p_y = (_y * 0.8 + 0.1) * tf.math.log(prob + 1e-5) + \
                         (1 - (_y * 0.8 + 0.1)) * tf.math.log(1 - prob + 1e-5)
                elb = tf.reduce_logsumexp(
                    log_p_y + self.beta * sg(log_p_z - log_q_z), -1
                ) - tf.math.log(float(sample_num))
                elb = tf.reduce_mean(elb)
                loss = -elb - r_elb
                
            else:
                # loss_type == 2: 更复杂的建模
                t = z @ self.wt1 + self.bt1
                t = tf.nn.tanh(t)
                t = t @ self.wt2 + self.bt2
                t = tf.squeeze(t, -1)
                
                s_y = z @ self.wsy1 + self.bsy1
                s_y = tf.nn.tanh(s_y)
                s_y = s_y @ self.wsy2 + self.bsy2
                s_y = afunc(s_y)
                s_y = tf.squeeze(s_y, -1)
                
                mu_y = t + tf.math.log((1e-5 + prob) / (1e-5 + 1 - prob)) * s_y
                y_pred = tf.reduce_mean(mu_y, -1)
                
                tmp = tf.nn.sigmoid((tf.expand_dims(y, -1) - mu_y) / s_y)
                log_p_y = tf.math.log((1e-5 + tmp) * (1e-5 + 1 - tmp) / s_y)
                elb = tf.reduce_logsumexp(
                    log_p_y + self.beta * sg(log_p_z - log_q_z), -1
                ) - tf.math.log(float(sample_num))
                elb = tf.reduce_mean(elb)
                loss = -elb - r_elb
            
            return y_pred, loss
