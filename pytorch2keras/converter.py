"""
The Pytorch2Keras converter module over JIT-trace.
"""

import torch
import torch.jit
import torch.autograd
import torch.serialization
import contextlib
from torch.jit import _unique_state_dict

from .layers import AVAILABLE_CONVERTERS


@contextlib.contextmanager
def set_training(model, mode):
    """
    A context manager to temporarily set the training mode of 'model'
    to 'mode', resetting it when we exit the with-block.  A no-op if
    mode is None.
    """
    if mode is None:
        yield
        return
    old_mode = model.training
    if old_mode != mode:
        model.train(mode)
    try:
        yield
    finally:
        if old_mode != mode:
            model.train(old_mode)


def _optimize_graph(graph, aten):
    # run dce first to eliminate dead parts of the graph that might have been
    # left behind by things like symbolic_override
    torch._C._jit_pass_dce(graph)
    torch._C._jit_pass_lint(graph)

    torch._C._jit_pass_peephole(graph)
    torch._C._jit_pass_lint(graph)
    graph = torch._C._jit_pass_onnx(graph, aten)
    torch._C._jit_pass_lint(graph)
    torch._C._jit_pass_onnx_peephole(graph)
    torch._C._jit_pass_lint(graph)
    torch._C._jit_pass_dce(graph)
    torch._C._jit_pass_lint(graph)
    graph = torch._C._jit_pass_canonicalize(graph)
    torch._C._jit_pass_lint(graph)
    return graph


def get_node_id(node):
    import re
    node_id = re.search(r"[\d]+", node.__str__())
    return node_id.group(0)


def pytorch_to_keras(
    model, args, input_shapes,
    change_ordering=False, training=False, verbose=False, short_names=False,
):
    """
    By given pytorch model convert layers with specified convertors.

    Args:
        model: pytorch model
        args: pytorch model arguments
        input_shapes: keras input shapes (using for each InputLayer)
        change_ordering: change CHW to HWC
        training: switch model to training mode
        verbose: verbose output
        short_names: use shorn names for keras layers

    Returns:
        model: created keras model.
    """

    # PyTorch JIT tracing
    if isinstance(args, torch.autograd.Variable):
        args = (args, )

    # Workaround for previous versions
    if isinstance(input_shapes, tuple):
        input_shapes = [input_shapes]

    orig_state_dict_keys = _unique_state_dict(model).keys()

    with set_training(model, training):
        trace, torch_out = torch.jit.get_trace_graph(model, tuple(args))

    if orig_state_dict_keys != _unique_state_dict(model).keys():
        raise RuntimeError("state_dict changed after running the tracer; "
                           "something weird is happening in your model!")

    # _optimize_trace(trace, False)
    trace.set_graph(_optimize_graph(trace.graph(), False))

    if verbose:
        print(trace.graph())

    if verbose:
        print(list(trace.graph().outputs()))

    # Get all graph nodes
    nodes = list(trace.graph().nodes())

    # Collect graph outputs
    graph_outputs = [n.uniqueName() for n in trace.graph().outputs()]
    print('Graph outputs:', graph_outputs)

    # Collect model state dict
    state_dict = _unique_state_dict(model)
    if verbose:
        print('State dict:', list(state_dict))

    import re
    import keras
    from keras import backend as K
    K.set_image_data_format('channels_first')

    layers = dict()
    keras_inputs = []
    for i in range(len(args)):
        layers['input{0}'.format(i)] = keras.layers.InputLayer(
            input_shape=input_shapes[i], name='input{0}'.format(i)
        ).output
        keras_inputs.append(layers['input{0}'.format(i)])

    outputs = []

    input_index = 0
    model_inputs = dict()
    for node in nodes:
        node_inputs = list(node.inputs())
        node_input_names = []
        for node_input in node_inputs:
            if node_input.node().scopeName():
                node_input_names.append(get_node_id(node_input.node()))

        if len(node_input_names) == 0:
            if len(node_inputs) > 0:
                if node_inputs[0] in model_inputs:
                    node_input_names.append(model_inputs[node_inputs[0]])
                else:
                    input_name = 'input{0}'.format(input_index)
                    node_input_names.append(input_name)
                    input_index += 1
                    model_inputs[node_inputs[0]] = input_name

        node_type = node.kind()
        # print(dir(node))

        node_scope_name = node.scopeName()
        node_id = get_node_id(node)
        node_weights_name = '.'.join(
            re.findall(r'\[([\w\d.]+)\]', node_scope_name)
        )
        node_attrs = {k: node[k] for k in node.attributeNames()}

        node_outputs = list(node.outputs())
        node_outputs_names = []
        for node_output in node_outputs:
            if node_output.node().scopeName():
                node_outputs_names.append(node_output.node().scopeName())

        if verbose:
            print(' ____ ')
            print('graph node:', node_scope_name)
            print('type:', node_type)
            print('inputs:', node_input_names)
            print('outputs:', node_outputs_names)
            print('name in state_dict:', node_weights_name)
            print('attrs:', node_attrs)
            print('is_terminal:', node_id in graph_outputs)
        AVAILABLE_CONVERTERS[node_type](
            node_attrs,
            node_weights_name, node_id,
            node_input_names,
            layers, state_dict,
            short_names
        )
        if node_id in graph_outputs:
            outputs.append(layers[node_id])

    model = keras.models.Model(inputs=keras_inputs, outputs=outputs)

    if change_ordering:
        import numpy as np
        conf = model.get_config()

        for layer in conf['layers']:
            if layer['config'] and 'batch_input_shape' in layer['config']:
                layer['config']['batch_input_shape'] = \
                    tuple(np.reshape(np.array(
                        [
                            [None] +
                            list(layer['config']['batch_input_shape'][2:][:]) +
                            [layer['config']['batch_input_shape'][1]]
                        ]), -1
                    ))
            if layer['config'] and 'target_shape' in layer['config']:
                if len(list(layer['config']['target_shape'][1:][:])) > 0:
                    layer['config']['target_shape'] = \
                        tuple(np.reshape(np.array(
                            [
                                list(layer['config']['target_shape'][1:][:]),
                                layer['config']['target_shape'][0]
                            ]), -1
                        ),)

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
