import keras.layers
import numpy as np
import random
import string
import tensorflow as tf
from .common import random_string


def convert_gemm(params, w_name, scope_name, inputs, layers, weights, names):
    """
    Convert Linear.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        names: use short names for keras layers
    """
    print('Converting Linear ...')

    if names == 'short':
        tf_name = 'FC' + random_string(6)
    elif names == 'keep':
        tf_name = w_name
    else:
        tf_name = w_name + str(random.random())

    bias_name = '{0}.bias'.format(w_name)
    weights_name = '{0}.weight'.format(w_name)

    W = weights[weights_name].numpy().transpose()
    input_channels, output_channels = W.shape

    keras_weights = [W]
    has_bias = False
    if bias_name in weights:
        bias = weights[bias_name].numpy()
        keras_weights = [W, bias]
        has_bias = True

    dense = keras.layers.Dense(
        output_channels,
        weights=keras_weights, use_bias=has_bias, name=tf_name, bias_initializer='zeros', kernel_initializer='zeros',
    )

    layers[scope_name] = dense(layers[inputs[0]])


def convert_matmul(params, w_name, scope_name, inputs, layers, weights, names):
    """
    Convert matmul layer.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        names: use short names for keras layers
    """
    print('Converting matmul ...')

    if names == 'short':
        tf_name = 'MMUL' + random_string(4)
    elif names == 'keep':
        tf_name = w_name
    else:
        tf_name = w_name + str(random.random())

    if len(inputs) == 1:
        weights_name = '{0}.weight'.format(w_name)

        W = weights[weights_name].numpy().transpose()
        input_channels, output_channels = W.shape

        keras_weights = [W]

        dense = keras.layers.Dense(
            output_channels,
            weights=keras_weights, use_bias=False, name=tf_name, bias_initializer='zeros', kernel_initializer='zeros',
        )
        layers[scope_name] = dense(layers[inputs[0]])
    elif len(inputs) == 2:
        weights_name = '{0}.weight'.format(w_name)

        W = weights[weights_name].numpy().transpose()
        input_channels, output_channels = W.shape

        keras_weights = [W]

        dense = keras.layers.Dense(
            output_channels,
            weights=keras_weights, use_bias=False, name=tf_name, bias_initializer='zeros', kernel_initializer='zeros',
        )
        layers[scope_name] = dense(layers[inputs[0]])
    else:
        raise AssertionError('Cannot convert matmul layer')
