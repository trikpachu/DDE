'''
Some functions imported by other methods.
TODO: Make also get_knn available this way.
'''

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as l

def fully_connected(inputs,
                    num_outputs,
                    use_xavier=True,
                    weight_decay=0.0,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bias=True):
    """ Fully connected layer with non-linear operation.

    Args:
      inputs: 2-D tensor BxN
      num_outputs: int
      use_xavier: bool, use xavier_initializer (i.e. glorot_normal) if true
      weight_decay: float
      activation_fn: function
      bn: bool, whether to use batch norm
      bn_decay: float or float tensor variable in [0,1]
    Returns:
      Variable tensor of size B x num_outputs.
    """

    if use_xavier:
        kernel_initializer='glorot_normal'
    else:
        kernel_initializer='glorot_uniform'

    outputs = l.Dense(
    num_outputs, activation=None, use_bias=bias, kernel_initializer=kernel_initializer,
    bias_initializer='zeros', kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay), bias_regularizer=None,
    activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(inputs)

    if bn:
        outputs = l.BatchNormalization(
        axis=-1, momentum=0.9, 
        epsilon=0.001, center=True, scale=True,
        beta_initializer='zeros', gamma_initializer='ones',
        moving_mean_initializer='zeros', moving_variance_initializer='ones',
        beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
        gamma_constraint=None, renorm=False, renorm_clipping=None, renorm_momentum=0.99,
        fused=None, trainable=True, virtual_batch_size=None, adjustment=None)(outputs)

    if activation_fn is not None:
        outputs = l.Activation(activation_fn)(outputs)

    return outputs

def mean_squared_error(y_true, y_pred):
    return np.average(((y_true - y_pred)) ** 2, axis=0)

def relative_mean_squared_error(y_true, y_pred):
    return np.average(((y_true - y_pred) / y_true) ** 2, axis=0)

def kl_divergence(estimate, true_dist):
    '''
    calculates kl divergence, add epsilon to zeros to get arround divisions by 0.
    '''
    epsilon = 1e-8
    estimate[np.where(estimate==0)] += epsilon
    true_dist[np.where(true_dist==0)] += epsilon
    true_dist = true_dist/np.sum(true_dist)
    estimate = estimate/np.sum(estimate)
    return np.sum(estimate*np.log(estimate/true_dist))


