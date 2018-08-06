import keras.layers
import numpy as np
import random
import string
import tensorflow as tf


def random_string(length):
    """
    Generate a random string for the layer name.
    :param length: a length of required random string
    :return: generated random string
    """
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))


def convert_conv(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert convolution layer.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
        short_names: use short names
    """
    print('Converting convolution ...')

    if short_names:
        tf_name = 'C' + random_string(7)
    else:
        tf_name = w_name + str(random.random())

    bias_name = '{0}.bias'.format(w_name)
    weights_name = '{0}.weight'.format(w_name)
    input_name = inputs[0]

    if len(weights[weights_name].numpy().shape) == 5: # 3D conv
        W = weights[weights_name].numpy().transpose(2, 3, 4, 1, 0)
        height, width, channels, n_layers, n_filters = W.shape
        print(W.shape)

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

        print(len(weights), len(weights[0]), len(weights[0][0]),
              len(weights[0][0][0]), len(weights[0][0][0][0]),
              len(weights[0][0][0][0][0]))
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
            name=tf_name
        )
        layers[scope_name] = conv(layers[input_name])
    elif len(weights[weights_name].numpy().shape) == 4:  # 2D conv
        W = weights[weights_name].numpy().transpose(2, 3, 1, 0)
        height, width, channels, n_filters = W.shape

        if params['group'] == n_filters:
            print('Perform depthwise convolution')

            if params['pads'][0] > 0 or params['pads'][1] > 0:
                padding_name = tf_name + '_pad'
                padding_layer = keras.layers.ZeroPadding2D(
                    padding=(params['pads'][0], params['pads'][1]),
                    name=padding_name
                )
                layers[padding_name] = padding_layer(layers[input_name])
                input_name = padding_name

            def target_layer(x):
                x = tf.transpose(x, [0, 2, 3, 1])

                # tensorflow.python.framework.errors_impl.UnimplementedError:
                # Depthwise convolution on CPU is only supported for NHWC format
                layer = tf.nn.depthwise_conv2d(x, W.transpose(0, 1, 3, 2),
                                               strides=(1, params['strides'][0], params['strides'][1], 1),
                                               padding='VALID', rate=[1, 1])
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

            if params['pads'][0] > 0 or params['pads'][1] > 0:
                padding_name = tf_name + '_pad'
                padding_layer = keras.layers.ZeroPadding2D(
                    padding=(params['pads'][0], params['pads'][1]),
                    name=padding_name
                )
                layers[padding_name] = padding_layer(layers[input_name])
                input_name = padding_name

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
    else:  # 1D conv
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


def convert_convtranspose(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert transposed convolution layer.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting transposed convolution ...')

    if short_names:
        tf_name = 'C' + random_string(7)
    else:
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

        assert(params['pads'][0] == 0)
        assert(params['pads'][1] == 0)

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
            weights=weights,
            use_bias=has_bias,
            activation=None,
            dilation_rate=params['dilations'][0],
            name=tf_name
        )
        layers[scope_name] = conv(layers[input_name])
    else:
        raise AssertionError('Layer is not supported for now')


def convert_flatten(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert reshape(view).

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting flatten ...')

    if short_names:
        tf_name = 'R' + random_string(7)
    else:
        tf_name = w_name + str(random.random())

    reshape = keras.layers.Flatten(name=tf_name)
    layers[scope_name] = reshape(layers[inputs[0]])


def convert_gemm(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert Linear.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting Linear ...')

    if short_names:
        tf_name = 'FC' + random_string(6)
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


def convert_avgpool(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert Average pooling.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting pooling ...')

    if short_names:
        tf_name = 'P' + random_string(7)
    else:
        tf_name = w_name + str(random.random())

    height, width = params['kernel_shape']
    stride_height, stride_width = params['strides']
    padding_h, padding_w, _, _ = params['pads']

    input_name = inputs[0]
    padding = 'valid'
    if padding_h > 0 and padding_w > 0:
        if padding_h == height // 2 and padding_w == width // 2:
            padding = 'same'
        else:
            raise AssertionError('Custom padding isnt supported')

    pooling = keras.layers.AveragePooling2D(
        pool_size=(height, width),
        strides=(stride_height, stride_width),
        padding=padding,
        name=tf_name
    )

    layers[scope_name] = pooling(layers[input_name])


def convert_maxpool(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert Max pooling.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """

    print('Converting pooling ...')

    if short_names:
        tf_name = 'P' + random_string(7)
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


def convert_maxpool3(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert 3d Max pooling.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """

    print('Converting pooling ...')

    if short_names:
        tf_name = 'P' + random_string(7)
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


def convert_dropout(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert dropout.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting dropout ...')

    if short_names:
        tf_name = 'DO' + random_string(6)
    else:
        tf_name = w_name + str(random.random())

    dropout = keras.layers.Dropout(rate=params['ratio'], name=tf_name)
    layers[scope_name] = dropout(layers[inputs[0]])


def convert_batchnorm(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert batch normalization layer.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting batchnorm ...')

    if short_names:
        tf_name = 'BN' + random_string(6)
    else:
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


def convert_elementwise_add(
    params, w_name, scope_name, inputs, layers, weights, short_names
):
    """
    Convert elementwise addition.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting elementwise_add ...')
    model0 = layers[inputs[0]]
    model1 = layers[inputs[1]]

    if short_names:
        tf_name = 'A' + random_string(7)
    else:
        tf_name = w_name + str(random.random())

    add = keras.layers.Add(name=tf_name)
    layers[scope_name] = add([model0, model1])


def convert_elementwise_mul(
    params, w_name, scope_name, inputs, layers, weights, short_names
):
    """
    Convert elementwise multiplication.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting elementwise_mul ...')
    model0 = layers[inputs[0]]
    model1 = layers[inputs[1]]

    if short_names:
        tf_name = 'M' + random_string(7)
    else:
        tf_name = w_name + str(random.random())

    mul = keras.layers.Multiply(name=tf_name)
    layers[scope_name] = mul([model0, model1])


def convert_elementwise_sub(
    params, w_name, scope_name, inputs, layers, weights, short_names
):
    """
    Convert elementwise subtraction.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting elementwise_sub ...')
    model0 = layers[inputs[0]]
    model1 = layers[inputs[1]]

    if short_names:
        tf_name = 'S' + random_string(7)
    else:
        tf_name = w_name + str(random.random())

    sub = keras.layers.Subtract(name=tf_name)
    layers[scope_name] = sub([model0, model1])


def convert_sum(
    params, w_name, scope_name, inputs, layers, weights, short_names
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
        short_names: use short names for keras layers
    """
    print('Converting Sum ...')

    def target_layer(x):
        import keras.backend as K
        return K.sum(x)

    lambda_layer = keras.layers.Lambda(target_layer)
    layers[scope_name] = lambda_layer(layers[inputs[0]])


def convert_concat(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert concatenation.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting concat ...')
    concat_nodes = [layers[i] for i in inputs]
    
    if short_names:
        tf_name = 'CAT' + random_string(5)
    else:
        tf_name = w_name + str(random.random())

    cat = keras.layers.Concatenate(name=tf_name, axis=params['axis'])
    layers[scope_name] = cat(concat_nodes)


def convert_relu(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert relu layer.

   Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting relu ...')

    if short_names:
        tf_name = 'RELU' + random_string(4)
    else:
        tf_name = w_name + str(random.random())

    relu = keras.layers.Activation('relu', name=tf_name)
    layers[scope_name] = relu(layers[inputs[0]])


def convert_lrelu(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert leaky relu layer.

   Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting lrelu ...')

    if short_names:
        tf_name = 'lRELU' + random_string(3)
    else:
        tf_name = w_name + str(random.random())

    leakyrelu = \
        keras.layers.LeakyReLU(alpha=params['alpha'], name=tf_name)
    layers[scope_name] = leakyrelu(layers[inputs[0]])


def convert_sigmoid(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert sigmoid layer.

   Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting sigmoid ...')

    if short_names:
        tf_name = 'SIGM' + random_string(4)
    else:
        tf_name = w_name + str(random.random())

    sigmoid = keras.layers.Activation('sigmoid', name=tf_name)
    layers[scope_name] = sigmoid(layers[inputs[0]])


def convert_softmax(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert softmax layer.

   Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting softmax ...')

    if short_names:
        tf_name = 'SMAX' + random_string(4)
    else:
        tf_name = w_name + str(random.random())

    softmax = keras.layers.Activation('softmax', name=tf_name)
    layers[scope_name] = softmax(layers[inputs[0]])


def convert_tanh(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert tanh layer.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting tanh ...')

    if short_names:
        tf_name = 'TANH' + random_string(4)
    else:
        tf_name = w_name + str(random.random())

    tanh = keras.layers.Activation('tanh', name=tf_name)
    layers[scope_name] = tanh(layers[inputs[0]])


def convert_hardtanh(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert hardtanh layer.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting hardtanh (clip) ...')

    def target_layer(x, max_val=float(params['max_val']), min_val=float(params['min_val'])):
        return tf.minimum(max_val, tf.maximum(min_val, x))

    lambda_layer = keras.layers.Lambda(target_layer)
    layers[scope_name] = lambda_layer(layers[inputs[0]])


def convert_selu(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert selu layer.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting selu ...')

    if short_names:
        tf_name = 'SELU' + random_string(4)
    else:
        tf_name = w_name + str(random.random())

    selu = keras.layers.Activation('selu', name=tf_name)
    layers[scope_name] = selu(layers[inputs[0]])


def convert_transpose(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert transpose layer.

   Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting transpose ...')
    if params['perm'][0] != 0:
        # raise AssertionError('Cannot permute batch dimension')
        print('!!! Cannot permute batch dimension. Result may be wrong !!!')
        try:
            layers[scope_name] = layers[inputs[0]]
        except:
            pass
    else:
        if short_names:
            tf_name = 'PERM' + random_string(4)
        else:
            tf_name = w_name + str(random.random())
        permute = keras.layers.Permute(params['perm'][1:], name=tf_name)
        layers[scope_name] = permute(layers[inputs[0]])


def convert_reshape(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert reshape layer.

   Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting reshape ...')
    if short_names:
        tf_name = 'RESH' + random_string(4)
    else:
        tf_name = w_name + str(random.random())

    print(layers[inputs[1]])
    if len(inputs) > 1:
        if layers[inputs[1]][0] == -1:
            print('Cannot deduct batch size! It will be omitted, but result may be wrong.')

        reshape = keras.layers.Reshape(layers[inputs[1]][1:], name=tf_name)
        layers[scope_name] = reshape(layers[inputs[0]])
    else:
        reshape = keras.layers.Reshape(params['shape'][1:], name=tf_name)
        layers[scope_name] = reshape(layers[inputs[0]])


def convert_matmul(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert matmul layer.

   Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting matmul ...')

    if short_names:
        tf_name = 'MMUL' + random_string(4)
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


def convert_gather(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert gather (embedding) layer.

   Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting embedding ...')

    if short_names:
        tf_name = 'EMBD' + random_string(4)
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
    layers[scope_name] = dense(layers[inputs[0]])


def convert_reduce_sum(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert reduce_sum layer.

   Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting reduce_sum ...')

    keepdims = params['keepdims'] > 0
    axis = params['axes']

    def target_layer(x, keepdims=keepdims, axis=axis):
        import keras.backend as K
        return K.sum(x, keepdims=keepdims, axis=axis)

    lambda_layer = keras.layers.Lambda(target_layer)
    layers[scope_name] = lambda_layer(layers[inputs[0]])


def convert_constant(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert constant layer.

   Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting constant ...')

    params_list = params['value'].numpy().tolist()

    def target_layer(x):
        import keras.backend as K
        return K.constant(params_list)

    lambda_layer = keras.layers.Lambda(target_layer)
    layers[scope_name] = lambda_layer(layers['input0']) # Temporary fix for nonexistent input name created by converter.py
    # layers[scope_name] = params['value']


def convert_upsample(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert upsample_bilinear2d layer.

   Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting upsample...')

    if params['mode'] != 'nearest':
        raise AssertionError('Cannot convert non-nearest upsampling')

    if short_names:
        tf_name = 'UPSL' + random_string(4)
    else:
        tf_name = w_name + str(random.random())

    scale = (params['height_scale'], params['width_scale'])
    upsampling = keras.layers.UpSampling2D(
        size=scale, name=tf_name
    )
    layers[scope_name] = upsampling(layers[inputs[0]])


def convert_padding(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert padding layer.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting padding...')

    if params['mode'] != 'constant':
        raise AssertionError('Cannot convert non-constant padding')

    if params['value'] != 0.0:
        raise AssertionError('Cannot convert non-zero padding')

    if short_names:
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


def convert_adaptive_avg_pool2d(params, w_name, scope_name, inputs, layers, weights, short_names):
    """
    Convert adaptive_avg_pool2d layer.

    Args:
        params: dictionary with layer parameters
        w_name: name prefix in state_dict
        scope_name: pytorch scope name
        inputs: pytorch node inputs
        layers: dictionary with keras tensors
        weights: pytorch state_dict
        short_names: use short names for keras layers
    """
    print('Converting adaptive_avg_pool2d...')

    if short_names:
        tf_name = 'APOL' + random_string(4)
    else:
        tf_name = w_name + str(random.random())

    global_pool = keras.layers.GlobalAveragePooling2D()
    layers_global_pool = global_pool(layers[inputs[0]])

    def target_layer(x):
        return keras.backend.expand_dims(x)

    lambda_layer = keras.layers.Lambda(target_layer)
    layers[scope_name] = lambda_layer(layers_global_pool)


AVAILABLE_CONVERTERS = {
    'onnx::Conv': convert_conv,
    'onnx::ConvTranspose': convert_convtranspose,
    'onnx::Flatten': convert_flatten,
    'onnx::Gemm': convert_gemm,
    'onnx::MaxPool': convert_maxpool,
    'max_pool2d': convert_maxpool,
    'aten::max_pool3d': convert_maxpool3,
    'onnx::AveragePool': convert_avgpool,
    'onnx::Dropout': convert_dropout,
    'onnx::BatchNormalization': convert_batchnorm,
    'onnx::Add': convert_elementwise_add,
    'onnx::Mul': convert_elementwise_mul,
    'onnx::Sub': convert_elementwise_sub,
    'onnx::Sum': convert_sum,
    'onnx::Concat': convert_concat,
    'onnx::Relu': convert_relu,
    'onnx::LeakyRelu': convert_lrelu,
    'onnx::Sigmoid': convert_sigmoid,
    'onnx::Softmax': convert_softmax,
    'onnx::Tanh': convert_tanh,
    'aten::hardtanh': convert_hardtanh,
    'onnx::Selu': convert_selu,
    'onnx::Transpose': convert_transpose,
    'onnx::Reshape': convert_reshape,
    'onnx::MatMul': convert_matmul,
    'onnx::Gather': convert_gather,
    'onnx::ReduceSum': convert_reduce_sum,
    'onnx::Constant': convert_constant,
    'onnx::Upsample': convert_upsample,
    'onnx::Pad': convert_padding,
    'aten::adaptive_avg_pool2d': convert_adaptive_avg_pool2d,
}
