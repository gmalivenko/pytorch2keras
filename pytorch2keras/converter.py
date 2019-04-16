"""
The PyTorch2Keras converter module over JIT-trace.
"""

import contextlib
from packaging import version
from collections import defaultdict
import six

import torch
import torch.jit
import torch.autograd
import torch.serialization
from torch.jit import _unique_state_dict

if version.parse('0.4.1') <= version.parse(torch.__version__):
    from torch.onnx import OperatorExportTypes

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


if version.parse('1.0.0') <= version.parse(torch.__version__):
    from torch._C import ListType

    # ONNX can't handle constants that are lists of tensors, which can
    # get generated in constant prop. So we split them back into prim::ListConstructs
    def _split_tensor_list_constants(g, block):
        for node in block.nodes():
            for subblock in node.blocks():
                _split_tensor_list_constants(g, subblock)
            if node.kind() == "prim::Constant":
                output_type = node.output().type()
                if output_type.isSubtypeOf(ListType.ofTensors()):
                    inputs = [g.create("prim::Constant").t_('value', t)
                               .insertBefore(node).output()
                              for t in node['value']]
                    lc = (g.create("prim::ListConstruct", inputs)
                          .insertBefore(node)
                          .output()
                          .setType(ListType.ofTensors()))
                    node.output().replaceAllUsesWith(lc)

if version.parse('1.0.0') > version.parse(torch.__version__):
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
else:
    def _optimize_graph(graph, operator_export_type=OperatorExportTypes.RAW):
        if version.parse('1.0.0') <= version.parse(torch.__version__):
            torch._C._jit_pass_remove_inplace_ops(graph)
            # we record now record some ops like ones/zeros
            # into a trace where we previously recorded constants
            # use constant prop to maintain our current level of onnx support
            # without implementing symbolics for all of them
            torch._C._jit_pass_constant_propagation(graph)
            # _split_tensor_list_constants(graph, graph)
            # run dce to eliminate dead parts of the graph that might have been
            # left behind by things like symbolic_override
            torch._C._jit_pass_dce(graph)
            torch._C._jit_pass_lint(graph)
            torch._C._jit_pass_canonicalize_ops(graph)
            torch._C._jit_pass_lint(graph)

            torch._C._jit_pass_peephole(graph, True)
            torch._C._jit_pass_lint(graph)

            # onnx only supports tensors, but 1 / 2 = 0.5 and tensor(1) / tensor(2) = 0
            torch._C._jit_pass_prepare_division_for_onnx(graph)
            # onnx only supports tensors, so we turn all out number types into tensors
            torch._C._jit_pass_erase_number_types(graph)
            # onnx does not support tuples, so try to remove them
            torch._C._jit_pass_lower_all_tuples(graph)
            # torch._C._jit_pass_peephole(graph, True)
            torch._C._jit_pass_lint(graph)

            if operator_export_type != OperatorExportTypes.RAW:
                graph = torch._C._jit_pass_onnx(graph, operator_export_type)
                torch._C._jit_pass_lint(graph)
                torch._C._jit_pass_onnx_peephole(graph)
                torch._C._jit_pass_lint(graph)
            torch._C._jit_pass_dce(graph)
            torch._C._jit_pass_lint(graph)
        else:
            torch._C._jit_pass_dce(graph)
            torch._C._jit_pass_lint(graph)

            torch._C._jit_pass_peephole(graph)
            torch._C._jit_pass_lint(graph)

            # torch._C._jit_pass_peephole(graph, True)
            torch._C._jit_pass_lint(graph)

            if operator_export_type != OperatorExportTypes.RAW:
                graph = torch._C._jit_pass_onnx(graph, operator_export_type)
                torch._C._jit_pass_lint(graph)
                torch._C._jit_pass_onnx_peephole(graph)
                torch._C._jit_pass_lint(graph)
            torch._C._jit_pass_dce(graph)
            torch._C._jit_pass_lint(graph)
        return graph


def get_node_id(node):
    import re
    node_id = re.search(r"[\d]+", node.__str__())
    return node_id.group(0)


def get_leaf_id(node, state={}):
    return str(node.uniqueName())


def pytorch_to_keras(
    model, args, input_shapes,
    change_ordering=False, training=False, verbose=False, names=False,
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
        names: use short names, use random-suffix or keep original names for keras layers

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
    if version.parse('0.4.0') < version.parse(torch.__version__):
        trace.set_graph(_optimize_graph(trace.graph(), OperatorExportTypes.ONNX))
    else:
        trace.set_graph(_optimize_graph(trace.graph(), False))

    trace.graph().lint()

    if verbose:
        print(trace.graph())

    # Get all graph nodes
    nodes = list(trace.graph().nodes())

    # Optimize Flatten:
    # When we have something loke that:
    #
    # %523 : Long() = onnx::Constant[value={0}](), scope: ResNet
    # %524 : Dynamic = onnx::Shape(%522), scope: ResNet
    # %526 : Long() = onnx::Gather[axis=0](%524, %523), scope: ResNet
    # %527 : Long() = onnx::Constant[value={-1}](), scope: ResNet
    # %534 : Dynamic = onnx::Unsqueeze[axes=[0]](%526)
    # %535 : Dynamic = onnx::Unsqueeze[axes=[0]](%527)
    # %536 : Dynamic = onnx::Concat[axis=0](%534, %535)
    # %529 : Float(1, 512) = onnx::Reshape(%522, %536), scope: ResNet
    #
    # It's better to replace it with onnx::Flatten
    if six.PY3:
        from types import SimpleNamespace
        seq_to_find = \
            ['onnx::Constant', 'onnx::Shape', 'onnx::Gather',
             'onnx::Constant', 'onnx::Unsqueeze', 'onnx::Unsqueeze', 'onnx::Concat', 'onnx::Reshape']
        k = 0
        s = 0
        for i, node in enumerate(nodes):
            if node.kind() == seq_to_find[k]:
                if k == 0:
                    s = i
                k += 1
                if k == len(seq_to_find):
                    reshape_op = nodes[s + k - 1]
                    flatten_op = {
                        'kind': (lambda: 'onnx::Flatten'),
                        'attributeNames': (lambda: {}),
                        'outputs':  (lambda: list(reshape_op.outputs())),
                        'scopeName': (lambda: reshape_op.scopeName()),
                        'inputs': (lambda: list(reshape_op.inputs())[:1]),
                        '__str__': (lambda: reshape_op.__str__()),
                    }
                    nodes = nodes[:s] + [SimpleNamespace(**flatten_op)] + nodes[s+k:]
                    break
            else:
                k = 0
                s = -1

    # Collect graph inputs and outputs
    graph_outputs = [get_leaf_id(n) for n in trace.graph().outputs()]
    graph_inputs = [get_leaf_id(n) for n in trace.graph().inputs()]

    # Collect model state dict
    state_dict = _unique_state_dict(model)
    if verbose:
        print('Graph inputs:', graph_inputs)
        print('Graph outputs:', graph_outputs)
        print('State dict:', list(state_dict))

    import re
    import keras
    from keras import backend as K
    K.set_image_data_format('channels_first')

    layers = dict()
    keras_inputs = []
    for i in range(len(args)):
        layers[graph_inputs[i]] = keras.layers.InputLayer(
            input_shape=input_shapes[i], name='input{0}'.format(i)
        ).output
        keras_inputs.append(layers[graph_inputs[i]])

    outputs = []
    group_indices = defaultdict(lambda: 0, {})

    for node in nodes:
        node_inputs = list(node.inputs())
        node_input_names = []

        for node_input in node_inputs:
            node_input_names.append(get_leaf_id(node_input))

        node_type = node.kind()

        node_scope_name = node.scopeName()
        node_id = get_node_id(node)
        node_name_regex = re.findall(r'\[([\w\d.\-\[\]\s]+)\]', node_scope_name)

        try: 
            int(node_name_regex[-1])
            node_weigth_group_name = '.'.join(
                node_name_regex[:-1]
            )
            node_weights_name = node_weigth_group_name + '.' + str(group_indices[node_weigth_group_name])
            group_indices[node_weigth_group_name] += 1

        except ValueError:
            node_weights_name = '.'.join(
                node_name_regex
            )
        except IndexError:
            node_weights_name = '.'.join(node_input_names)

        node_attrs = {k: node[k] for k in node.attributeNames()}

        node_outputs = list(node.outputs())
        node_outputs_names = []
        for node_output in node_outputs:
            if node_output.node().scopeName():
                node_outputs_names.append(node_output.node().scopeName())

        if verbose:
            print(' ____ ')
            print('graph node:', node_scope_name)
            print('node id:', node_id)
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
            names
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

    print('Your model was (probably) successfully converted! '
          'Please, follow the repository https://github.com/nerox8664/pytorch2keras and give a star :)')
    return model
