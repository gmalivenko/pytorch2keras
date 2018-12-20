import keras.layers
import numpy as np
import random
import string
import tensorflow as tf
from .common import random_string


def convert_avgpool(params, w_name, scope_name, inputs, layers, weights, names):
    """
    Convert Average pooling.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        names: use short names for keras layers
    """
    print('Converting pooling ...')

    if names == 'short':
        tf_name = 'P' + random_string(7)
    elif names == 'keep':
        tf_name = w_name
    else:
        tf_name = w_name + str(random.random())

    if 'kernel_shape' in params:
        height, width = params['kernel_shape']
    else:
        height, width = params['kernel_size']

    if 'strides' in params:
        stride_height, stride_width = params['strides']
    else:
        stride_height, stride_width = params['stride']

    if 'pads' in params:
        padding_h, padding_w, _, _ = params['pads']
    else:
        padding_h, padding_w = params['padding']

    input_name = inputs[0]
    pad = 'valid' 

    if height % 2 == 1 and width % 2 == 1 and \
       height // 2 == padding_h and width // 2 == padding_w and \
       stride_height == 1 and stride_width == 1:
        pad = 'same'
    else:
        padding_name = tf_name + '_pad'
        padding_layer = keras.layers.ZeroPadding2D(
            padding=(padding_h, padding_w),
            name=padding_name
        )
        layers[padding_name] = padding_layer(layers[inputs[0]])
        input_name = padding_name

    # Pooling type AveragePooling2D
    pooling = keras.layers.AveragePooling2D(
        pool_size=(height, width),
        strides=(stride_height, stride_width),
        padding=pad,
        name=tf_name,
        data_format='channels_first'
    )

    layers[scope_name] = pooling(layers[input_name])


def convert_maxpool(params, w_name, scope_name, inputs, layers, weights, names):
    """
    Convert Max pooling.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        names: use short names for keras layers
    """

    print('Converting pooling ...')

    if names == 'short':
        tf_name = 'P' + random_string(7)
    elif names == 'keep':
        tf_name = w_name
    else:
        tf_name = w_name + str(random.random())

    if 'kernel_shape' in params:
        height, width = params['kernel_shape']
    else:
        height, width = params['kernel_size']

    if 'strides' in params:
        stride_height, stride_width = params['strides']
    else:
        stride_height, stride_width = params['stride']

    if 'pads' in params:
        padding_h, padding_w, _, _ = params['pads']
    else:
        padding_h, padding_w = params['padding']

    input_name = inputs[0]
    pad = 'valid' 

    if height % 2 == 1 and width % 2 == 1 and \
       height // 2 == padding_h and width // 2 == padding_w and \
       stride_height == 1 and stride_width == 1:
        pad = 'same'
    else:
        padding_name = tf_name + '_pad'
        padding_layer = keras.layers.ZeroPadding2D(
            padding=(padding_h, padding_w),
            name=padding_name
        )
        layers[padding_name] = padding_layer(layers[inputs[0]])
        input_name = padding_name

    # Pooling type MaxPooling2D
    pooling = keras.layers.MaxPooling2D(
        pool_size=(height, width),
        strides=(stride_height, stride_width),
        padding=pad,
        name=tf_name,
        data_format='channels_first'
    )

    layers[scope_name] = pooling(layers[input_name])


def convert_maxpool3(params, w_name, scope_name, inputs, layers, weights, names):
    """
    Convert 3d Max pooling.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        names: use short names for keras layers
    """

    print('Converting pooling ...')

    if names == 'short':
        tf_name = 'P' + random_string(7)
    elif names == 'keep':
        tf_name = w_name
    else:
        tf_name = w_name + str(random.random())

    if 'kernel_shape' in params:
        height, width, depth = params['kernel_shape']
    else:
        height, width, depth = params['kernel_size']

    if 'strides' in params:
        stride_height, stride_width, stride_depth = params['strides']
    else:
        stride_height, stride_width, stride_depth = params['stride']

    if 'pads' in params:
        padding_h, padding_w, padding_d, _, _ = params['pads']
    else:
        padding_h, padding_w, padding_d = params['padding']

    input_name = inputs[0]
    if padding_h > 0 and padding_w > 0 and padding_d > 0:
        padding_name = tf_name + '_pad'
        padding_layer = keras.layers.ZeroPadding3D(
            padding=(padding_h, padding_w, padding_d),
            name=padding_name
        )
        layers[padding_name] = padding_layer(layers[inputs[0]])
        input_name = padding_name

    # Pooling type
    pooling = keras.layers.MaxPooling3D(
        pool_size=(height, width, depth),
        strides=(stride_height, stride_width, stride_depth),
        padding='valid',
        name=tf_name
    )

    layers[scope_name] = pooling(layers[input_name])


def convert_adaptive_avg_pool2d(params, w_name, scope_name, inputs, layers, weights, names):
    """
    Convert adaptive_avg_pool2d layer.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        names: use short names for keras layers
    """
    print('Converting adaptive_avg_pool2d...')

    if names == 'short':
        tf_name = 'APOL' + random_string(4)
    elif names == 'keep':
        tf_name = w_name
    else:
        tf_name = w_name + str(random.random())

    global_pool = keras.layers.GlobalAveragePooling2D(data_format='channels_first', name=tf_name)
    layers[scope_name] = global_pool(layers[inputs[0]])

    def target_layer(x):
        import keras
        return keras.backend.expand_dims(x)

    lambda_layer = keras.layers.Lambda(target_layer, name=tf_name + 'E')
    layers[scope_name] = lambda_layer(layers[scope_name])  # double expand dims
    layers[scope_name] = lambda_layer(layers[scope_name])


def convert_adaptive_max_pool2d(params, w_name, scope_name, inputs, layers, weights, names):
    """
    Convert convert_adaptive_max_pool2d layer.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        names: use short names for keras layers
    """
    print('Converting adaptive_avg_pool2d...')

    if names == 'short':
        tf_name = 'APOL' + random_string(4)
    elif names == 'keep':
        tf_name = w_name
    else:
        tf_name = w_name + str(random.random())

    global_pool = keras.layers.GlobalMaxPooling2D(data_format='channels_first', name=tf_name)
    layers[scope_name] = global_pool(layers[inputs[0]])

    def target_layer(x):
        import keras
        return keras.backend.expand_dims(x)

    lambda_layer = keras.layers.Lambda(target_layer, name=tf_name + 'E')
    layers[scope_name] = lambda_layer(layers[scope_name])  # double expand dims
    layers[scope_name] = lambda_layer(layers[scope_name])
