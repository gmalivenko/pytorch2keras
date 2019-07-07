"""
The PyTorch2Keras converter interface
"""

from onnx2keras import onnx_to_keras
import torch
import onnx
from onnx import optimizer
import io
import logging


def pytorch_to_keras(
    model, args, input_shapes=None,
    change_ordering=False, verbose=False, name_policy=None,
    use_optimizer=False, do_constant_folding=False
):
    """
    By given PyTorch model convert layers with ONNX.

    Args:
        model: pytorch model
        args: pytorch model arguments
        input_shapes: keras input shapes (using for each InputLayer)
        change_ordering: change CHW to HWC
        verbose: verbose output
        name_policy: use short names, use random-suffix or keep original names for keras layers

    Returns:
        model: created keras model.
    """
    logger = logging.getLogger('pytorch2keras')

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    logger.info('Converter is called.')

    if name_policy:
        logger.warning('Name policy isn\'t supported now.')

    if input_shapes:
        logger.warning('Custom shapes isn\'t supported now.')

    if input_shapes and not isinstance(input_shapes, list):
        input_shapes = [input_shapes]

    if not isinstance(args, list):
        args = [args]

    args = tuple(args)

    dummy_output = model(*args)

    if isinstance(dummy_output, torch.autograd.Variable):
        dummy_output = [dummy_output]

    input_names = ['input_{0}'.format(i) for i in range(len(args))]
    output_names = ['output_{0}'.format(i) for i in range(len(dummy_output))]

    logger.debug('Input_names:')
    logger.debug(input_names)

    logger.debug('Output_names:')
    logger.debug(output_names)

    stream = io.BytesIO()
    torch.onnx.export(model, args, stream, do_constant_folding=do_constant_folding, verbose=verbose, input_names=input_names, output_names=output_names)

    stream.seek(0)
    onnx_model = onnx.load(stream)
    if use_optimizer:
        if use_optimizer is True:
            optimizer2run = optimizer.get_available_passes()
        else:
            use_optimizer = set(use_optimizer)
            optimizer2run = [x for x in optimizer.get_available_passes() if x in use_optimizer]
        logger.info("Running optimizer:\n%s", "\n".join(optimizer2run))
        onnx_model = optimizer.optimize(onnx_model, optimizer2run)

    k_model = onnx_to_keras(onnx_model=onnx_model, input_names=input_names,
                            input_shapes=input_shapes, name_policy=name_policy,
                            verbose=verbose, change_ordering=change_ordering)

    return k_model
