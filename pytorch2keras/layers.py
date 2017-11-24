import numpy as np
import keras.layers


def convert_dense(node, node_name, input_name, output_name, layers):
    """
    Convert fully-connected.

    Args:
        node: pytorch node element.
        node_name: pytorch node name
        input_name: pytorch input node name
        output_name: pytorch output node name
        layers: dictionary with keras tensors
    """
    print('Conerting dense ...')
    W = \
        node.next_functions[2][0].next_functions[0][0].variable.data.numpy()
    output_channels, input_channels = W.shape
    W = np.transpose(W)
    weights = [W]

    if node.next_functions[0][0]:
        bias = node.next_functions[0][0].variable.data.numpy()
        has_bias = True
        weights = [W, bias]
    else:
        has_bias = False

    dense = keras.layers.Dense(
        output_channels,
        weights=weights, use_bias=has_bias, name=output_name
    )
    layers[output_name] = dense(layers[input_name])


def convert_reshape(node, node_name, input_name, output_name, layers):
    """
    Convert reshape(view).

    Args:
        node: pytorch node element.
        node_name: pytorch node name
        input_name: pytorch input node name
        output_name: pytorch output node name
        layers: dictionary with keras tensors
    """
    print('Conerting reshape ...')
    target_shape = node.new_sizes
    reshape = keras.layers.Reshape(target_shape[1:], name=output_name)
    layers[output_name] = reshape(layers[input_name])


def convert_convolution(node, node_name, input_name, output_name, layers):
    """
    Convert convolution layer.

    Args:
        node: pytorch node element.
        node_name: pytorch node name
        input_name: pytorch input node name
        output_name: pytorch output node name
        layers: dictionary with keras tensors
    """
    print('Conerting convolution ...')
    weights_var = node.next_functions[1][0].variable

    if node.next_functions[2][0]:
        biases = node.next_functions[2][0].variable.data.numpy()
        has_bias = True
    else:
        biases = None
        has_bias = False

    if node.transposed:
        W = weights_var.data.numpy().transpose(2, 3, 1, 0)
        height, width, n_filters, channels = W.shape
    else:
        W = weights_var.data.numpy().transpose(2, 3, 1, 0)
        height, width, channels, n_filters = W.shape

    assert node.output_padding == (0, 0)

    if node.padding[0] != node.padding[1]:
        raise ValueError('Unsuported padding size for convolution')

    if node.dilation[0] != node.dilation[1]:
        raise ValueError('Unsuported dilation rate for convolution')
    dilation_rate = node.dilation[0]

    padding = node.padding[0]
    if padding > 0:
        padding_name = output_name + '_pad'
        padding_layer = keras.layers.ZeroPadding2D(
            padding=node.padding,
            name=padding_name
        )
        layers[padding_name] = padding_layer(layers[input_name])
        input_name = padding_name

    weights = None
    if has_bias:
        weights = [W, biases]
    else:
        weights = [W]

    border_mode = 'valid'
    # if padding == 1:
    #     border_mode = 'same'

    if node.transposed:
        conv = keras.layers.Conv2DTranspose(
            filters=n_filters,
            kernel_size=(height, width),
            strides=(node.stride[0], node.stride[1]),
            padding=border_mode,
            weights=weights,
            use_bias=has_bias,
            activation=None,
            dilation_rate=dilation_rate,
            name=output_name
        )
    else:
        conv = keras.layers.Conv2D(
            filters=n_filters,
            kernel_size=(height, width),
            strides=(node.stride[0], node.stride[1]),
            padding=border_mode,
            weights=weights,
            use_bias=has_bias,
            activation=None,
            dilation_rate=dilation_rate,
            name=output_name
        )
    layers[output_name] = conv(layers[input_name])


def convert_batchnorm(node, node_name, input_name, output_name, layers):
    """
    Convert batch normalization layer.

    Args:
        node: pytorch node element.
        node_name: pytorch node name
        input_name: pytorch input node name
        output_name: pytorch output node name
        layers: dictionary with keras tensors
    """
    print('Conerting batchnorm ...')

    weights = None
    biases = None
    if node.next_functions[1][0]:
        weights = node.next_functions[1][0].variable.data
        biases = node.next_functions[2][0].variable.data
        gamma = weights.numpy()
        beta = biases.numpy()

    mean = node.running_mean.numpy()
    variance = node.running_var.numpy()

    eps = node.eps
    momentum = node.momentum

    if weights is None:
        bn = keras.layers.BatchNormalization(
            axis=1, momentum=momentum, epsilon=eps,
            center=False, scale=False,
            weights=[mean, variance],
            name=output_name
        )
    else:
        bn = keras.layers.BatchNormalization(
            axis=1, momentum=momentum, epsilon=eps,
            weights=[gamma, beta, mean, variance],
            name=output_name
        )
    layers[output_name] = bn(layers[input_name])


def convert_pooling(node, node_name, input_name, output_name, layers):
    """
    Convert pooling layer.

    Args:
        node: pytorch node element.
        node_name: pytorch node name
        input_name: pytorch input node name
        output_name: pytorch output node name
        layers: dictionary with keras tensors
    """
    print('Conerting pooling ...')

    height, width = node.kernel_size
    stride_height, stride_width = node.stride

    if isinstance(node.padding, tuple) and node.padding[0] != node.padding[1]:
        raise ValueError('Unsuported padding size for pooling')

    if isinstance(node.padding, int):
        padding = node.padding
    else:
        padding = node.padding[0]

    border_mode = 'valid'
    # if padding == 1:
    #     border_mode = 'same'

    if padding > 0:
        padding_name = output_name + '_pad'
        padding_layer = keras.layers.ZeroPadding2D(
            padding=node.padding,
            name=padding_name
        )
        layers[padding_name] = padding_layer(layers[input_name])
        input_name = padding_name

    # Pooling type
    if node_name.startswith('Max'):
        pooling = keras.layers.MaxPooling2D(
            pool_size=node.kernel_size,
            strides=(stride_height, stride_width),
            padding=border_mode,
            name=output_name
        )
    elif node_name.startswith('Avg'):
        pooling = keras.layers.AveragePooling2D(
            pool_size=node.kernel_size,
            strides=(stride_height, stride_width),
            padding=border_mode,
            name=output_name
        )
    else:
        raise ValueError('Unknown pooling type')

    layers[output_name] = pooling(layers[input_name])


def convert_elementwise_add(node, node_name, input_names, output_name, layers):
    """
    Convert elementwise addition.

    Args:
        node: pytorch node element.
        node_name: pytorch node name
        input_name: pytorch input node name
        output_name: pytorch output node name
        layers: dictionary with keras tensors
    """
    print('Conerting elementwise_add ...')
    model0 = layers[input_names[0]]
    model1 = layers[input_names[1]]

    add = keras.layers.Add(name=output_name)
    layers[output_name] = add([model0, model1])


def convert_threshold(node, node_name, input_name, output_name, layers):
    """
    Convert relu activation.

    Args:
        node: pytorch node element.
        node_name: pytorch node name
        input_name: pytorch input node name
        output_name: pytorch output node name
        layers: dictionary with keras tensors
    """
    print('Conerting RELU ...')
    relu = keras.layers.Activation('relu', name=output_name)
    layers[output_name] = relu(layers[input_name])


def convert_leakyrelu(node, node_name, input_name, output_name, layers):
    """
    Convert relu activation.

    Args:
        node: pytorch node element.
        node_name: pytorch node name
        input_name: pytorch input node name
        output_name: pytorch output node name
        layers: dictionary with keras tensors
    """
    print('Conerting LeakyRELU ...')
    leakyrelu = \
        keras.layers.LeakyReLU(alpha=node.additional_args[0], name=output_name)
    layers[output_name] = leakyrelu(layers[input_name])


def convert_prelu(node, node_name, input_name, output_name, layers):
    """
    Convert PReLU activation.
    TODO: handle single-value tensor and raise `wrong shape` exception.
    Args:
        node: pytorch node element.
        node_name: pytorch node name
        input_name: pytorch input node name
        output_name: pytorch output node name
        layers: dictionary with keras tensors
    """
    print('Conerting PReLU ...')
    a = node.next_functions[1][0].variable.data.numpy()
    prelu = keras.layers.PReLU(name=output_name, weights=np.array([a]))
    layers[output_name] = prelu(layers[input_name])


def convert_selu(node, node_name, input_name, output_name, layers):
    """
    Convert selu activation.

    Args:
        node: pytorch node element.
        node_name: pytorch node name
        input_name: pytorch input node name
        output_name: pytorch output node name
        layers: dictionary with keras tensors
    """
    print('Conerting SELU ...')
    selu = keras.layers.Activation('selu', name=output_name)
    layers[output_name] = selu(layers[input_name])


def convert_tanh(node, node_name, input_name, output_name, layers):
    """
    Convert tanh activation.

    Args:
        node: pytorch node element.
        node_name: pytorch node name
        input_name: pytorch input node name
        output_name: pytorch output node name
        layers: dictionary with keras tensors
    """
    print('Conerting tanh ...')
    tanh = keras.layers.Activation('tanh', name=output_name)
    layers[output_name] = tanh(layers[input_name])


def convert_softmax(node, node_name, input_name, output_name, layers):
    """
    Convert softmax activation.

    Args:
        node: pytorch node element.
        node_name: pytorch node name
        input_name: pytorch input node name
        output_name: pytorch output node name
        layers: dictionary with keras tensors
    """
    print('Conerting softmax ...')
    softmax = keras.layers.Activation('softmax', name=output_name)
    layers[output_name] = softmax(layers[input_name])


def convert_softplus(node, node_name, input_name, output_name, layers):
    """
    Convert softplus activation.

    Args:
        node: pytorch node element.
        node_name: pytorch node name
        input_name: pytorch input node name
        output_name: pytorch output node name
        layers: dictionary with keras tensors
    """
    print('Conerting softplus ...')
    softplus = keras.layers.Activation('softplus', name=output_name)
    layers[output_name] = softplus(layers[input_name])


def convert_softsign(node, node_name, input_name, output_name, layers):
    """
    Convert softsign activation.

    Args:
        node: pytorch node element.
        node_name: pytorch node name
        input_name: pytorch input node name
        output_name: pytorch output node name
        layers: dictionary with keras tensors
    """
    print('Conerting softsign ...')
    softsign = keras.layers.Activation('softsign', name=output_name)
    layers[output_name] = softsign(layers[input_name])


def convert_sigmoid(node, node_name, input_name, output_name, layers):
    """
    Convert sigmoid activation.

    Args:
        node: pytorch node element.
        node_name: pytorch node name
        input_name: pytorch input node name
        output_name: pytorch output node name
        layers: dictionary with keras tensors
    """
    print('Conerting sigmoid ...')
    sigmoid = keras.layers.Activation('sigmoid', name=output_name)
    layers[output_name] = sigmoid(layers[input_name])


AVAILABLE_CONVERTERS = {
    'Addmm': convert_dense,
    'ConvNd': convert_convolution,
    'BatchNorm': convert_batchnorm,
    'MaxPool2d': convert_pooling,
    'AvgPool2d': convert_pooling,
    'Add': convert_elementwise_add,
    'View': convert_reshape,
    'Threshold': convert_threshold,
    'LeakyReLU': convert_leakyrelu,
    'PReLU': convert_prelu,
    'SELU': convert_selu,
    'Tanh': convert_tanh,
    'Softmax': convert_softmax,
    'Softplus': convert_softplus,
    'Softsign': convert_softsign,
    'Sigmoid': convert_sigmoid,
}
