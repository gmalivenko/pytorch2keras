import keras.layers
import numpy as np
import random
import string
import tensorflow as tf
from .common import random_string


def convert_relu(params, w_name, scope_name, inputs, layers, weights, names):
    """
    Convert relu layer.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        names: use short names for keras layers
    """
    print('Converting relu ...')

    if names == 'short':
        tf_name = 'RELU' + random_string(4)
    elif names == 'keep':
        tf_name = w_name
    else:
        tf_name = w_name + str(random.random())

    relu = keras.layers.Activation('relu', name=tf_name)
    layers[scope_name] = relu(layers[inputs[0]])


def convert_lrelu(params, w_name, scope_name, inputs, layers, weights, names):
    """
    Convert leaky relu layer.

   Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        names: use short names for keras layers
    """
    print('Converting lrelu ...')

    if names == 'short':
        tf_name = 'lRELU' + random_string(3)
    elif names == 'keep':
        tf_name = w_name
    else:
        tf_name = w_name + str(random.random())

    leakyrelu = \
        keras.layers.LeakyReLU(alpha=params['alpha'], name=tf_name)
    layers[scope_name] = leakyrelu(layers[inputs[0]])


def convert_sigmoid(params, w_name, scope_name, inputs, layers, weights, names):
    """
    Convert sigmoid layer.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        names: use short names for keras layers
    """
    print('Converting sigmoid ...')

    if names == 'short':
        tf_name = 'SIGM' + random_string(4)
    elif names == 'keep':
        tf_name = w_name
    else:
        tf_name = w_name + str(random.random())

    sigmoid = keras.layers.Activation('sigmoid', name=tf_name)
    layers[scope_name] = sigmoid(layers[inputs[0]])


def convert_softmax(params, w_name, scope_name, inputs, layers, weights, names):
    """
    Convert softmax layer.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        names: use short names for keras layers
    """
    print('Converting softmax ...')

    if names == 'short':
        tf_name = 'SMAX' + random_string(4)
    elif names == 'keep':
        tf_name = w_name
    else:
        tf_name = w_name + str(random.random())

    if 'axis' in params:
        axis = params['axis']
    if 'value' in params:
        axis = params['value'].item()
    else:
        if len(inputs) > 1:
            axis = layers[inputs[1] + '_np']

    def target_layer(x, dim=axis):
        import keras
        return keras.activations.softmax(x, axis=dim)

    lambda_layer = keras.layers.Lambda(target_layer)
    layers[scope_name] = lambda_layer(layers[inputs[0]])


def convert_tanh(params, w_name, scope_name, inputs, layers, weights, names):
    """
    Convert tanh layer.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        names: use short names for keras layers
    """
    print('Converting tanh ...')

    if names == 'short':
        tf_name = 'TANH' + random_string(4)
    elif names == 'keep':
        tf_name = w_name
    else:
        tf_name = w_name + str(random.random())

    tanh = keras.layers.Activation('tanh', name=tf_name)
    layers[scope_name] = tanh(layers[inputs[0]])


def convert_hardtanh(params, w_name, scope_name, inputs, layers, weights, names):
    """
    Convert hardtanh layer.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        names: use short names for keras layers
    """
    print('Converting hardtanh (clip) ...')

    def target_layer(x, max_val=float(params['max_val']), min_val=float(params['min_val'])):
        return tf.minimum(max_val, tf.maximum(min_val, x))

    lambda_layer = keras.layers.Lambda(target_layer)
    layers[scope_name] = lambda_layer(layers[inputs[0]])


def convert_selu(params, w_name, scope_name, inputs, layers, weights, names):
    """
    Convert selu layer.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        names: use short names for keras layers
    """
    print('Converting selu ...')

    if names == 'short':
        tf_name = 'SELU' + random_string(4)
    elif names == 'keep':
        tf_name = w_name
    else:
        tf_name = w_name + str(random.random())

    selu = keras.layers.Activation('selu', name=tf_name)
    layers[scope_name] = selu(layers[inputs[0]])
