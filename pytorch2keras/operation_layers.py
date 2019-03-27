import keras.layers
import numpy as np
import random
import string
import tensorflow as tf
from .common import random_string



def convert_sum(
    params, w_name, scope_name, inputs, layers, weights, names
):
    """
    Convert sum.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        names: use short names for keras layers
    """
    print('Converting Sum ...')

    def target_layer(x):
        import keras.backend as K
        return K.sum(x)

    lambda_layer = keras.layers.Lambda(target_layer)
    layers[scope_name] = lambda_layer(layers[inputs[0]])


def convert_reduce_sum(params, w_name, scope_name, inputs, layers, weights, names):
    """
    Convert reduce_sum layer.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        names: use short names for keras layers
    """
    print('Converting reduce_sum ...')

    keepdims = params['keepdims'] > 0
    axis = params['axes']

    def target_layer(x, keepdims=keepdims, axis=axis):
        import keras.backend as K
        return K.sum(x, keepdims=keepdims, axis=axis)

    lambda_layer = keras.layers.Lambda(target_layer)
    layers[scope_name] = lambda_layer(layers[inputs[0]])

def convert_concat(params, w_name, scope_name, inputs, layers, weights, names):
    """
    Convert concatenation.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        names: use short names for keras layers
    """
    print('Converting concat ...')
    concat_nodes = [layers[i] for i in inputs]

    if len(concat_nodes) == 1:
        # no-op
        layers[scope_name] = concat_nodes[0]
        return

    if names == 'short':
        tf_name = 'CAT' + random_string(5)
    elif names == 'keep':
        tf_name = w_name
    else:
        tf_name = w_name + str(random.random())

    cat = keras.layers.Concatenate(name=tf_name, axis=params['axis'])
    layers[scope_name] = cat(concat_nodes)


def convert_slice(params, w_name, scope_name, inputs, layers, weights, names):
    """
    Convert slice operation.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        names: use short names for keras layers
    """
    print('Converting slice ...')

    if len(params['axes']) > 1:
        raise AssertionError('Cannot convert slice by multiple dimensions')

    if params['axes'][0] not in [0, 1, 2, 3]:
        raise AssertionError('Slice by dimension more than 3 or less than 0 is not supported')

    def target_layer(x, axis=int(params['axes'][0]), start=int(params['starts'][0]), end=int(params['ends'][0])):
        if axis == 0:
            return x[start:end]
        elif axis == 1:
            return x[:, start:end]
        elif axis == 2:
            return x[:, :, start:end]
        elif axis == 3:
            return x[:, :, :, start:end]

    lambda_layer = keras.layers.Lambda(target_layer)
    layers[scope_name] = lambda_layer(layers[inputs[0]])


def convert_clip(params, w_name, scope_name, inputs, layers, weights, names):
    """
    Convert clip operation.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        names: use short names for keras layers
    """
    print('Converting clip ...')

    if params['min'] == 0:
        print("using ReLU({0})".format(params['max']))
        layer = keras.layers.ReLU(max_value=params['max'])
    else:
        def target_layer(x, vmin=params['min'], vmax=params['max']):
            import tensorflow as tf
            return tf.clip_by_value(x, vmin, vmax)
        layer = keras.layers.Lambda(target_layer)

    layers[scope_name] = layer(layers[inputs[0]])
