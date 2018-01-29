"""
The Pytorch2Keras converter module over JIT-trace.
"""

import torch
import torch.jit
import torch.autograd
import torch.serialization
import contextlib
from torch.jit import _unique_state_dict

from layers import AVAILABLE_CONVERTERS


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


def _optimize_trace(trace, aten):
    # run dce first to eliminate dead parts of the graph that might have been
    # left behind by things like symbolic_override
    torch._C._jit_pass_dce(trace)
    torch._C._jit_pass_lint(trace)

    torch._C._jit_pass_peephole(trace)
    torch._C._jit_pass_lint(trace)
    torch._C._jit_pass_onnx(trace, aten)
    torch._C._jit_pass_lint(trace)
    torch._C._jit_pass_onnx_peephole(trace)
    torch._C._jit_pass_lint(trace)
    torch._C._jit_pass_dce(trace)
    torch._C._jit_pass_lint(trace)
    torch._C._jit_pass_canonicalize(trace)
    torch._C._jit_pass_lint(trace)


def pytorch_to_keras(
    model, args, input_shape,
    change_ordering=False, training=False, verbose=False
):
    """
    By given pytorch model convert layers with specified convertors.

    Args:
        model: pytorch model
        args: pytorch model arguments
        input_shape: keras input shape (using for InputLayer creation)
        change_ordering: change CHW to HWC
        training: switch model to training mode
        verbose: verbose output

    Returns:
        model: created keras model.
    """

    # PyTorch JIT tracing
    if isinstance(args, torch.autograd.Variable):
        args = (args, )

    orig_state_dict_keys = _unique_state_dict(model).keys()

    with set_training(model, training):
        trace, torch_out = torch.jit.trace(model, args)

    if orig_state_dict_keys != _unique_state_dict(model).keys():
        raise RuntimeError("state_dict changed after running the tracer; "
                           "something weird is happening in your model!")

    _optimize_trace(trace, False)

    if verbose:
        print(list(trace.graph().outputs()))

    # Get all graph nodes
    nodes = list(trace.graph().nodes())

    # Collect graph outputs
    graph_outputs = [n.uniqueName() for n in trace.graph().outputs()]

    # Collect model state dict
    state_dict = _unique_state_dict(model)
    if verbose:
        print(list(state_dict))

    import re
    import keras
    from keras import backend as K
    K.set_image_data_format('channels_first')

    layers = dict()
    layers['input'] = keras.layers.InputLayer(
        input_shape=input_shape, name='input'
    ).output

    outputs = []

    for node in nodes:
        node_inputs = list(node.inputs())
        node_input_names = []
        for node_input in node_inputs:
            if node_input.node().scopeName():
                node_input_names.append(node_input.node().scopeName())

        if len(node_input_names) == 0:
            node_input_names.append('input')

        node_type = node.kind()
        node_scope_name = node.scopeName()
        node_id = re.search(r"[\d]+", node.__str__())[0]
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
            node_weights_name, node_scope_name,
            node_input_names,
            layers, state_dict
        )
        if node_id in graph_outputs:
            outputs.append(layers[node_scope_name])

    model = keras.models.Model(inputs=layers['input'], outputs=outputs)

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
