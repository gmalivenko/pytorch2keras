import keras
from collections import OrderedDict

from layers import AVAILABLE_CONVERTERS


"""
Pytorch to Keras converter.
"""


def collect_nodes(var):
    """
    Walk throught pytorch graph buttom-up and collects all the nodes.
    To avoid redundand computations TransposeBackward is skipped
    when follows AddmmBackward type.

    Args:
        var: pytorch variable that represents output of the model.

    Returns:
        nodes: list of functions sorted topologically from a root(s),
                input variable(s).

    """
    seen = set()
    nodes = OrderedDict()

    def add_node(func, nodes):
        parent_type = str(type(func).__name__)
        children = []

        if hasattr(func, 'next_functions'):
            for u in func.next_functions:
                next_func = u[0]
                if next_func is not None:
                    child_type = str(type(next_func).__name__)
                    if not (child_type == 'AccumulateGrad' or
                            parent_type == 'AddmmBackward' and
                            child_type == 'TransposeBackward'):
                        children.append(next_func)
                        if next_func not in seen:
                            seen.add(next_func)
                            add_node(next_func, nodes)

        nodes[func] = children

    add_node(var.grad_fn, nodes)

    return nodes


def create_keras_model(nodes, layers, input_node_name='input'):
    """
    By given nodes list convert layers with specified convertors.

    Args:
        nodes: list with pytorch nodes (OrderedDict)
        layers: dictionary with an input tensor
        input_node_name: name of the input tensor

    Returns:
        tensor: output keras tensor.
    """

    node_names = {}
    output_node_name = ''
    for i, node in enumerate(nodes):
        node_names[node] = \
            '{0}_{1}'.format(
                str(type(node).__name__.replace('Backward', '')),
                i)
        output_node_name = node_names[node]

    while nodes:
        node, children = nodes.popitem(last=False)
        node_type = str(type(node).__name__.replace('Backward', ''))
        node_name = node_names[node]

        print('Queued {0}, processing {1}'.format(len(nodes), node_name))

        if not children:
            input_name = input_node_name
        else:
            input_name = [node_names[child] for child in children]
            if len(input_name) == 1:
                input_name = input_name[0]

        output_name = node_name if nodes else output_node_name
        try:
            AVAILABLE_CONVERTERS[node_type](
                node, node_name, input_name, output_name, layers
            )
        except KeyError:
            print(
                'An error occured.'
                'The {0} can\'t be converted because'
                ' it is not currently supported.'.format(
                    node_type)
            )
            raise KeyError('Layer is not currently supported.')
    return layers[output_node_name]


def pytorch_to_keras(input_shape, pytorch_output, change_ordering=False):
    """
    By given nodes list convert layers with specified convertors.

    Args:
        input_shape: keras input shape (using for InputLayer creation)
        pytorch_output: putorch model output
        change_ordering: change CHW to HWC

    Returns:
        model: created keras model.
    """
    from keras import backend as K
    K.set_image_data_format('channels_first')
    nodes = collect_nodes(pytorch_output)

    layers = dict()
    layers['input'] = keras.layers.InputLayer(
        input_shape=input_shape, name='input'
    ).output

    output = create_keras_model(nodes, layers)
    model = keras.models.Model(inputs=layers['input'], outputs=output)

    if change_ordering:
        import numpy as np
        conf = model.get_config()
        for layer in conf['layers']:
            if layer['config'] and 'batch_input_shape' in layer['config']:
                layer['config']['batch_input_shape'] = \
                    tuple(np.reshape(
                        [
                            None,
                            *layer['config']['batch_input_shape'][2:][:],
                            layer['config']['batch_input_shape'][1]
                        ], -1
                    ))
            if layer['config'] and 'data_format' in layer['config']:
                layer['config']['data_format'] = 'channels_last'
            if layer['config'] and 'axis' in layer['config']:
                layer['config']['axis'] = 3

        K.set_image_data_format('channels_last')
        model_tf_ordering = keras.models.Model.from_config(conf)

        # from keras.utils.layer_utils import convert_all_kernels_in_model
        # convert_all_kernels_in_model(model)

        for dst_layer, src_layer in zip(
            model_tf_ordering.layers, model.layers
        ):
            dst_layer.set_weights(src_layer.get_weights())

        model = model_tf_ordering

    return model
