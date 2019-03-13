import keras.layers
import numpy as np
import random
import string
import tensorflow as tf
from .common import random_string


def convert_conv(params, w_name, scope_name, inputs, layers, weights, names):
    """
    Convert convolution layer.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        names: use short names for keras layers
    """
    print('Converting convolution ...')

    if names == 'short':
        tf_name = 'C' + random_string(7)
    elif names == 'keep':
        tf_name = w_name
    else:
        tf_name = w_name + str(random.random())

    bias_name = '{0}.bias'.format(w_name)
    weights_name = '{0}.weight'.format(w_name)
    input_name = inputs[0]

    if len(weights[weights_name].numpy().shape) == 5:  # 3D conv
        W = weights[weights_name].numpy().transpose(2, 3, 4, 1, 0)
        height, width, channels, n_layers, n_filters = W.shape

        if bias_name in weights:
            biases = weights[bias_name].numpy()
            has_bias = True
        else:
            biases = None
            has_bias = False

        if params['pads'][0] > 0 or params['pads'][1] > 0:
            padding_name = tf_name + '_pad'
            padding_layer = keras.layers.ZeroPadding3D(
                padding=(params['pads'][0],
                         params['pads'][1],
                         params['pads'][2]),
                name=padding_name
            )
            layers[padding_name] = padding_layer(layers[input_name])
            input_name = padding_name

        if has_bias:
            weights = [W, biases]
        else:
            weights = [W]

        conv = keras.layers.Conv3D(
            filters=n_filters,
            kernel_size=(channels, height, width),
            strides=(params['strides'][0],
                     params['strides'][1],
                     params['strides'][2]),
            padding='valid',
            weights=weights,
            use_bias=has_bias,
            activation=None,
            dilation_rate=params['dilations'][0],
            bias_initializer='zeros', kernel_initializer='zeros',
            name=tf_name
        )
        layers[scope_name] = conv(layers[input_name])

    elif len(weights[weights_name].numpy().shape) == 4:  # 2D conv
        if params['pads'][0] > 0 or params['pads'][1] > 0:
            padding_name = tf_name + '_pad'
            padding_layer = keras.layers.ZeroPadding2D(
                padding=(params['pads'][0], params['pads'][1]),
                name=padding_name
            )
            layers[padding_name] = padding_layer(layers[input_name])
            input_name = padding_name

        W = weights[weights_name].numpy().transpose(2, 3, 1, 0)
        height, width, channels_per_group, out_channels = W.shape
        n_groups = params['group']
        in_channels = channels_per_group * n_groups

        if n_groups == in_channels and n_groups != 1:
            if bias_name in weights:
                biases = weights[bias_name].numpy()
                has_bias = True
            else:
                biases = None
                has_bias = False

            W = W.transpose(0, 1, 3, 2)
            if has_bias:
                weights = [W, biases]
            else:
                weights = [W]

            conv = keras.layers.DepthwiseConv2D(
                kernel_size=(height, width),
                strides=(params['strides'][0], params['strides'][1]),
                padding='valid',
                use_bias=has_bias,
                activation=None,
                depth_multiplier=1,
                weights = weights,
                dilation_rate=params['dilations'][0],
                bias_initializer='zeros', kernel_initializer='zeros'
            )
            layers[scope_name] = conv(layers[input_name])

        elif n_groups != 1:
            # Example from https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html
            # # Split input and weights and convolve them separately
            # input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
            # weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
            # output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

            # # Concat the convolved output together again
            # conv = tf.concat(axis=3, values=output_groups)
            def target_layer(x, groups=params['group'], stride_y=params['strides'][0], stride_x=params['strides'][1]):
                x = tf.transpose(x, [0, 2, 3, 1])

                def convolve_lambda(i, k):
                    return tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1], padding='VALID')

                input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
                weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=W.transpose(0, 1, 2, 3))
                output_groups = [convolve_lambda(i, k) for i, k in zip(input_groups, weight_groups)]

                layer = tf.concat(axis=3, values=output_groups)

                layer = tf.transpose(layer, [0, 3, 1, 2])
                return layer

            lambda_layer = keras.layers.Lambda(target_layer)
            layers[scope_name] = lambda_layer(layers[input_name])

        else:
            if bias_name in weights:
                biases = weights[bias_name].numpy()
                has_bias = True
            else:
                biases = None
                has_bias = False

            if has_bias:
                weights = [W, biases]
            else:
                weights = [W]

            conv = keras.layers.Conv2D(
                filters=out_channels,
                kernel_size=(height, width),
                strides=(params['strides'][0], params['strides'][1]),
                padding='valid',
                weights=weights,
                use_bias=has_bias,
                activation=None,
                dilation_rate=params['dilations'][0],
                bias_initializer='zeros', kernel_initializer='zeros',
                name=tf_name
            )
            layers[scope_name] = conv(layers[input_name])

    else:  # 1D conv
        W = weights[weights_name].numpy().transpose(2, 1, 0)
        width, channels, n_filters = W.shape
        n_groups = params['group']
        if n_groups > 1:
            raise AssertionError('Cannot convert conv1d with groups != 1')

        if bias_name in weights:
            biases = weights[bias_name].numpy()
            has_bias = True
        else:
            biases = None
            has_bias = False

        padding_name = tf_name + '_pad'
        padding_layer = keras.layers.ZeroPadding1D(
            padding=params['pads'][0],
            name=padding_name
        )
        layers[padding_name] = padding_layer(layers[inputs[0]])
        input_name = padding_name

        if has_bias:
            weights = [W, biases]
        else:
            weights = [W]

        conv = keras.layers.Conv1D(
            filters=channels,
            kernel_size=width,
            strides=params['strides'],
            padding='valid',
            weights=weights,
            use_bias=has_bias,
            activation=None,
            data_format='channels_first',
            dilation_rate=params['dilations'],
            bias_initializer='zeros', kernel_initializer='zeros',
            name=tf_name
        )
        layers[scope_name] = conv(layers[input_name])


def convert_convtranspose(params, w_name, scope_name, inputs, layers, weights, names):
    """
    Convert transposed convolution layer.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        names: use short names for keras layers
    """
    print('Converting transposed convolution ...')

    if names == 'short':
        tf_name = 'C' + random_string(7)
    elif names == 'keep':
        tf_name = w_name
    else:
        tf_name = w_name + str(random.random())

    bias_name = '{0}.bias'.format(w_name)
    weights_name = '{0}.weight'.format(w_name)

    if len(weights[weights_name].numpy().shape) == 4:
        W = weights[weights_name].numpy().transpose(2, 3, 1, 0)
        height, width, n_filters, channels = W.shape

        n_groups = params['group']
        if n_groups > 1:
            raise AssertionError('Cannot convert conv1d with groups != 1')

        if params['dilations'][0] > 1:
            raise AssertionError('Cannot convert conv1d with dilation_rate != 1')

        if bias_name in weights:
            biases = weights[bias_name].numpy()
            has_bias = True
        else:
            biases = None
            has_bias = False

        input_name = inputs[0]

        if has_bias:
            weights = [W, biases]
        else:
            weights = [W]

        conv = keras.layers.Conv2DTranspose(
            filters=n_filters,
            kernel_size=(height, width),
            strides=(params['strides'][0], params['strides'][1]),
            padding='valid',
            output_padding=0,
            weights=weights,
            use_bias=has_bias,
            activation=None,
            dilation_rate=params['dilations'][0],
            bias_initializer='zeros', kernel_initializer='zeros',
            name=tf_name
        )

        layers[scope_name] = conv(layers[input_name])

        # Magic ad-hoc.
        # See the Keras issue: https://github.com/keras-team/keras/issues/6777
        layers[scope_name].set_shape(layers[scope_name]._keras_shape)

        pads = params['pads']
        if pads[0] > 0:
            assert(len(pads) == 2 or (pads[2] == pads[0] and pads[3] == pads[1]))

            crop = keras.layers.Cropping2D(
                pads[:2],
                name=tf_name + '_crop'
            )
            layers[scope_name] = crop(layers[scope_name])
    else:
        raise AssertionError('Layer is not supported for now')
