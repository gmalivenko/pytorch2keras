import keras.layers
import numpy as np
import random
import string
import tensorflow as tf
from .common import random_string


def convert_flatten(params, w_name, scope_name, inputs, layers, weights, names):
    """
    Convert reshape(view).

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        names: use short names for keras layers
    """
    print('Converting flatten ...')

    if names == 'short':
        tf_name = 'R' + random_string(7)
    elif names == 'keep':
        tf_name = w_name
    else:
        tf_name = w_name + str(random.random())

    reshape = keras.layers.Reshape([-1], name=tf_name)
    layers[scope_name] = reshape(layers[inputs[0]])


def convert_transpose(params, w_name, scope_name, inputs, layers, weights, names):
    """
    Convert transpose layer.

   Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        names: use short names for keras layers
    """
    print('Converting transpose ...')
    if params['perm'][0] != 0:
        if inputs[0] in layers:
            print('!!! Cannot permute batch dimension. Result may be wrong !!!')
            layers[scope_name] = layers[inputs[0]]
        else:
            print('Skip weight matrix transpose, result may be wrong.')
    else:
        if names:
            tf_name = 'PERM' + random_string(4)
        else:
            tf_name = w_name + str(random.random())
        permute = keras.layers.Permute(params['perm'][1:], name=tf_name)
        layers[scope_name] = permute(layers[inputs[0]])


def convert_reshape(params, w_name, scope_name, inputs, layers, weights, names):
    """
    Convert reshape layer.

   Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        names: use short names for keras layers
    """
    print('Converting reshape ...')
    if names == 'short':
        tf_name = 'RESH' + random_string(4)
    elif names == 'keep':
        tf_name = w_name
    else:
        tf_name = w_name + str(random.random())

    if len(inputs) > 1:
        if layers[inputs[1]][0] == -1:
            print('Cannot deduct batch size! It will be omitted, but result may be wrong.')

        reshape = keras.layers.Reshape(layers[inputs[1] + '_np'], name=tf_name)
        layers[scope_name] = reshape(layers[inputs[0]])
    else:
        if inputs[0] in layers:
            reshape = keras.layers.Reshape(params['shape'][1:], name=tf_name)
            layers[scope_name] = reshape(layers[inputs[0]])
        else:
            print('Skip weight matrix transpose, but result may be wrong.')

def convert_squeeze(params, w_name, scope_name, inputs, layers, weights, names):
    """
    Convert squeeze operation.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        names: use short names for keras layers
    """
    print('Converting squeeze ...')

    if len(params['axes']) > 1:
        raise AssertionError('Cannot convert squeeze by multiple dimensions')

    def target_layer(x, axis=int(params['axes'][0])):
        import tensorflow as tf
        return tf.squeeze(x, axis=axis)

    lambda_layer = keras.layers.Lambda(target_layer)
    layers[scope_name] = lambda_layer(layers[inputs[0]])


def convert_unsqueeze(params, w_name, scope_name, inputs, layers, weights, names):
    """
    Convert unsqueeze operation.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        names: use short names for keras layers
    """
    print('Converting unsqueeze ...')

    if names == 'short':
        tf_name = 'UNSQ' + random_string(4)
    elif names == 'keep':
        tf_name = w_name
    else:
        tf_name = w_name + str(random.random())

    def target_layer(x):
        import keras
        return keras.backend.expand_dims(x)

    lambda_layer = keras.layers.Lambda(target_layer, name=tf_name + 'E')
    layers[scope_name] = lambda_layer(layers[inputs[0]])


def convert_shape(params, w_name, scope_name, inputs, layers, weights, names):
    """
    Convert shape operation.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        names: use short names for keras layers
    """
    print('Converting shape ...')

    def target_layer(x):
        import tensorflow as tf
        return tf.shape(x)

    lambda_layer = keras.layers.Lambda(target_layer)
    layers[scope_name] = lambda_layer(layers[inputs[0]])