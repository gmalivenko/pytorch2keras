import keras.layers
import random


def convert_conv(params, w_name, scope_name, inputs, layers, weights):
    """
    Convert convolution layer.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
    """
    print('Converting convolution ...')

    tf_name = w_name + str(random.random())
    bias_name = '{0}.bias'.format(w_name)
    weights_name = '{0}.weight'.format(w_name)

    if len(weights[weights_name].numpy().shape) == 4:
        W = weights[weights_name].numpy().transpose(2, 3, 1, 0)
        height, width, channels, n_filters = W.shape

        if bias_name in weights:
            biases = weights[bias_name].numpy()
            has_bias = True
        else:
            biases = None
            has_bias = False

        padding_name = tf_name + '_pad'
        padding_layer = keras.layers.ZeroPadding2D(
            padding=(params['pads'][0], params['pads'][1]),
            name=padding_name
        )
        layers[padding_name] = padding_layer(layers[inputs[0]])
        input_name = padding_name

        weights = None
        if has_bias:
            weights = [W, biases]
        else:
            weights = [W]

        conv = keras.layers.Conv2D(
            filters=n_filters,
            kernel_size=(height, width),
            strides=(params['strides'][0], params['strides'][1]),
            padding='valid',
            weights=weights,
            use_bias=has_bias,
            activation=None,
            dilation_rate=params['dilations'][0],
            name=tf_name
        )
        layers[scope_name] = conv(layers[input_name])
    else:
        W = weights[weights_name].numpy().transpose(2, 1, 0)
        width, channels, n_filters = W.shape

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

        weights = None
        if has_bias:
            weights = [W, biases]
        else:
            weights = [W]

        conv = keras.layers.Conv1D(
            filters=n_filters,
            kernel_size=width,
            strides=params['strides'][0],
            padding='valid',
            weights=weights,
            use_bias=has_bias,
            activation=None,
            dilation_rate=params['dilations'][0],
            name=tf_name
        )
        layers[scope_name] = conv(layers[input_name])


def convert_convtranspose(params, w_name, scope_name, inputs, layers, weights):
    """
    Convert transposed convolution layer.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
    """
    print('Converting transposed convolution ...')

    tf_name = w_name + str(random.random())
    bias_name = '{0}.bias'.format(w_name)
    weights_name = '{0}.weight'.format(w_name)

    if len(weights[weights_name].numpy().shape) == 4:
        W = weights[weights_name].numpy().transpose(2, 3, 1, 0)
        height, width, n_filters, channels = W.shape

        if bias_name in weights:
            biases = weights[bias_name].numpy()
            has_bias = True
        else:
            biases = None
            has_bias = False

        padding_name = tf_name + '_pad'
        padding_layer = keras.layers.ZeroPadding2D(
            padding=(params['pads'][0], params['pads'][1]),
            name=padding_name
        )
        layers[padding_name] = padding_layer(layers[inputs[0]])
        input_name = padding_name

        weights = None
        if has_bias:
            weights = [W, biases]
        else:
            weights = [W]

        conv = keras.layers.Conv2DTranspose(
            filters=n_filters,
            kernel_size=(height, width),
            strides=(params['strides'][0], params['strides'][1]),
            padding='valid',
            weights=weights,
            use_bias=has_bias,
            activation=None,
            dilation_rate=params['dilations'][0],
            name=tf_name
        )
        layers[scope_name] = conv(layers[input_name])
    else:
        raise AssertionError('Layer is not supported for now')


def convert_flatten(params, w_name, scope_name, inputs, layers, weights):
    """
    Convert reshape(view).

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
    """
    print('Conerting reshape ...')
    tf_name = w_name + str(random.random())
    reshape = keras.layers.Flatten(name=tf_name)
    layers[scope_name] = reshape(layers[inputs[0]])


def convert_gemm(params, w_name, scope_name, inputs, layers, weights):
    """
    Convert Linear.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
    """
    print('Converting Linear ...')

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
        weights=keras_weights, use_bias=has_bias, name=tf_name
    )

    layers[scope_name] = dense(layers[inputs[0]])


def convert_avgpool(params, w_name, scope_name, inputs, layers, weights):
    """
    Convert Average pooling.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
    """
    print('Converting pooling ...')

    tf_name = w_name + str(random.random())
    height, width = params['kernel_shape']
    stride_height, stride_width = params['strides']
    padding_h, padding_w, _, _ = params['pads']

    input_name = inputs[0]
    if padding_h > 0 and padding_w > 0:
        padding_name = tf_name + '_pad'
        padding_layer = keras.layers.ZeroPadding2D(
            padding=(padding_h, padding_w),
            name=padding_name
        )
        layers[padding_name] = padding_layer(layers[inputs[0]])
        input_name = padding_name

    # Pooling type
    pooling = keras.layers.AveragePooling2D(
        pool_size=(height, width),
        strides=(stride_height, stride_width),
        padding='valid',
        name=tf_name
    )

    layers[scope_name] = pooling(layers[input_name])


def convert_maxpool(params, w_name, scope_name, inputs, layers, weights):
    """
    Convert Max pooling.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
    """

    print('Converting pooling ...')

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
    if padding_h > 0 and padding_w > 0:
        padding_name = tf_name + '_pad'
        padding_layer = keras.layers.ZeroPadding2D(
            padding=(padding_h, padding_w),
            name=padding_name
        )
        layers[padding_name] = padding_layer(layers[inputs[0]])
        input_name = padding_name

    # Pooling type
    pooling = keras.layers.MaxPooling2D(
        pool_size=(height, width),
        strides=(stride_height, stride_width),
        padding='valid',
        name=tf_name
    )

    layers[scope_name] = pooling(layers[input_name])


def convert_dropout(params, w_name, scope_name, inputs, layers, weights):
    """
    Convert dropout.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
    """
    print('Converting dropout ...')

    tf_name = w_name + str(random.random())
    dropout = keras.layers.Dropout(rate=params['ratio'], name=tf_name)
    layers[scope_name] = dropout(layers[inputs[0]])


def convert_batchnorm(params, w_name, scope_name, inputs, layers, weights):
    """
    Convert batch normalization layer.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
    """
    print('Converting batchnorm ...')

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


def convert_elementwise_add(params, w_name, scope_name, inputs, layers, weights):
    """
    Convert elementwise addition.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
    """
    print('Converting elementwise_add ...')
    model0 = layers[inputs[0]]
    model1 = layers[inputs[1]]

    tf_name = w_name + str(random.random())

    add = keras.layers.Add(name=tf_name)
    layers[scope_name] = add([model0, model1])


def convert_elementwise_mul(params, w_name, scope_name, inputs, layers, weights):
    """
    Convert elementwise multiplication.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
    """
    print('Converting elementwise_mul ...')
    model0 = layers[inputs[0]]
    model1 = layers[inputs[1]]

    tf_name = w_name + str(random.random())

    mul = keras.layers.Multiply(name=tf_name)
    layers[scope_name] = mul([model0, model1])


def convert_elementwise_sub(params, w_name, scope_name, inputs, layers, weights):
    """
    Convert elementwise subtraction.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
    """
    print('Converting elementwise_sub ...')
    model0 = layers[inputs[0]]
    model1 = layers[inputs[1]]

    tf_name = w_name + str(random.random())

    sub = keras.layers.Subtract(name=tf_name)
    layers[scope_name] = sub([model0, model1])


def convert_concat(params, w_name, scope_name, inputs, layers, weights):
    """
    Convert concatenation.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
    """
    print('Converting concat ...')
    concat_nodes = [layers[i] for i in inputs]
    tf_name = w_name + str(random.random())
    cat = keras.layers.Concatenate(name=tf_name, axis=params['axis'])
    layers[scope_name] = cat(concat_nodes)


def convert_relu(params, w_name, scope_name, inputs, layers, weights):
    """
    Convert relu layer.

   Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
    """
    print('Converting relu ...')

    tf_name = w_name + str(random.random())
    relu = keras.layers.Activation('relu', name=tf_name)
    layers[scope_name] = relu(layers[inputs[0]])


def convert_sigmoid(params, w_name, scope_name, inputs, layers, weights):
    """
    Convert sigmoid layer.

   Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
    """
    print('Converting sigmoid ...')

    tf_name = w_name + str(random.random())
    sigmoid = keras.layers.Activation('sigmoid', name=tf_name)
    layers[scope_name] = sigmoid(layers[inputs[0]])


def convert_softmax(params, w_name, scope_name, inputs, layers, weights):
    """
    Convert softmax layer.

   Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
    """
    print('Converting softmax ...')

    tf_name = w_name + str(random.random())
    softmax = keras.layers.Activation('softmax', name=tf_name)
    layers[scope_name] = softmax(layers[inputs[0]])


def convert_tanh(params, w_name, scope_name, inputs, layers, weights):
    """
    Convert tanh layer.

   Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
    """
    print('Converting tanh ...')

    tf_name = w_name + str(random.random())
    tanh = keras.layers.Activation('tanh', name=tf_name)
    layers[scope_name] = tanh(layers[inputs[0]])


AVAILABLE_CONVERTERS = {
    'Conv': convert_conv,
    'ConvTranspose': convert_convtranspose,
    'Flatten': convert_flatten,
    'Gemm': convert_gemm,
    'MaxPool': convert_maxpool,
    'max_pool2d': convert_maxpool,
    'AveragePool': convert_avgpool,
    'Dropout': convert_dropout,
    'BatchNormalization': convert_batchnorm,
    'Add': convert_elementwise_add,
    'Mul': convert_elementwise_mul,
    'Sub': convert_elementwise_sub,
    'Concat': convert_concat,
    'Relu': convert_relu,
    'Sigmoid': convert_sigmoid,
    'Softmax': convert_softmax,
    'Tanh': convert_tanh,
}
