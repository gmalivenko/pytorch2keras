import keras.layers
import numpy as np
import random
import string
import tensorflow as tf
from .common import random_string


def convert_padding(params, w_name, scope_name, inputs, layers, weights, names):
    """
    Convert padding layer.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        names: use short names for keras layers
    """
    print('Converting padding...')

    if params['mode'] == 'constant':
        # raise AssertionError('Cannot convert non-constant padding')

        if params['value'] != 0.0:
            raise AssertionError('Cannot convert non-zero padding')

        if names:
            tf_name = 'PADD' + random_string(4)
        else:
            tf_name = w_name + str(random.random())

        # Magic ordering
        padding_name = tf_name
        padding_layer = keras.layers.ZeroPadding2D(
            padding=((params['pads'][2], params['pads'][6]), (params['pads'][3], params['pads'][7])),
            name=padding_name
        )

        layers[scope_name] = padding_layer(layers[inputs[0]])
    elif params['mode'] == 'reflect':

        def target_layer(x, pads=params['pads']):
            # x = tf.transpose(x, [0, 2, 3, 1])
            layer = tf.pad(x, [[0, 0], [0, 0], [pads[2], pads[6]], [pads[3], pads[7]]], 'REFLECT')
            # layer = tf.transpose(layer, [0, 3, 1, 2])
            return layer

        lambda_layer = keras.layers.Lambda(target_layer)
        layers[scope_name] = lambda_layer(layers[inputs[0]])
