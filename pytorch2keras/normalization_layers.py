import keras.layers
import numpy as np
import random
import string
import tensorflow as tf
from .common import random_string


def convert_batchnorm(params, w_name, scope_name, inputs, layers, weights, names):
    """
    Convert batch normalization layer.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        names: use short names for keras layers
    """
    print('Converting batchnorm ...')

    if names == 'short':
        tf_name = 'BN' + random_string(6)
    elif names == 'keep':
        tf_name = w_name
    else:
        tf_name = w_name + str(random.random())

    bias_name = '{0}.bias'.format(w_name)
    weights_name = '{0}.weight'.format(w_name)
    mean_name = '{0}.running_mean'.format(w_name)
    var_name = '{0}.running_var'.format(w_name)

    if bias_name in weights:
        beta = weights[bias_name].numpy()

    if weights_name in weights:
        gamma = weights[weights_name].numpy()

    mean = weights[mean_name].numpy()
    variance = weights[var_name].numpy()

    eps = params['epsilon']
    momentum = params['momentum']

    if weights_name not in weights:
        bn = keras.layers.BatchNormalization(
            axis=1, momentum=momentum, epsilon=eps,
            center=False, scale=False,
            weights=[mean, variance],
            name=tf_name
        )
    else:
        bn = keras.layers.BatchNormalization(
            axis=1, momentum=momentum, epsilon=eps,
            weights=[gamma, beta, mean, variance],
            name=tf_name
        )
    layers[scope_name] = bn(layers[inputs[0]])


def convert_instancenorm(params, w_name, scope_name, inputs, layers, weights, names):
    """
    Convert instance normalization layer.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        names: use short names for keras layers
    """
    print('Converting instancenorm ...')

    if names == 'short':
        tf_name = 'IN' + random_string(6)
    elif names == 'keep':
        tf_name = w_name
    else:
        tf_name = w_name + str(random.random())

    assert(len(inputs) == 3)

    bias_name = '{0}.bias'.format(w_name)
    weights_name = '{0}.weight'.format(w_name)

    # Use previously taken constants
    if inputs[-2] + '_np' in layers:
        gamma = layers[inputs[-2] + '_np']
    else:
        gamma = weights[weights_name].numpy()

    if inputs[-1] + '_np' in layers:
        beta = layers[inputs[-1] + '_np']
    else:
        beta = weights[bias_name].numpy()

    def target_layer(x, epsilon=params['epsilon'], gamma=gamma, beta=beta):
        layer = tf.contrib.layers.instance_norm(
            x,
            param_initializers={'beta': tf.constant_initializer(beta), 'gamma': tf.constant_initializer(gamma)},
            epsilon=epsilon, data_format='NCHW',
            trainable=False
        )
        return layer

    lambda_layer = keras.layers.Lambda(target_layer, name=tf_name)
    layers[scope_name] = lambda_layer(layers[inputs[0]])


def convert_dropout(params, w_name, scope_name, inputs, layers, weights, names):
    """
    Convert dropout.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        names: use short names for keras layers
    """
    print('Converting dropout ...')

    if names == 'short':
        tf_name = 'DO' + random_string(6)
    elif names == 'keep':
        tf_name = w_name
    else:
        tf_name = w_name + str(random.random())

    dropout = keras.layers.Dropout(rate=params['ratio'], name=tf_name)
    layers[scope_name] = dropout(layers[inputs[0]])