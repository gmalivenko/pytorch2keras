import keras.layers
import numpy as np
import random
import string
import tensorflow as tf
from .common import random_string


def convert_gather(params, w_name, scope_name, inputs, layers, weights, names):
    """
    Convert gather (embedding) layer.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        names: use short names for keras layers
    """
    print('Converting embedding ...')

    if names == 'short':
        tf_name = 'EMBD' + random_string(4)
    elif names == 'keep':
        tf_name = w_name
    else:
        tf_name = w_name + str(random.random())

    weights_name = '{0}.weight'.format(w_name)

    W = weights[weights_name].numpy()
    input_channels, output_channels = W.shape

    keras_weights = [W]

    dense = keras.layers.Embedding(
        input_channels,
        weights=keras_weights, output_dim=output_channels, name=tf_name
    )
    layers[scope_name] = dense(layers[inputs[1]])
