import keras.layers
import numpy as np
import random
import string
import tensorflow as tf


from .convolution_layers import convert_conv, convert_convtranspose
from .reshape_layers import convert_flatten, convert_transpose, convert_reshape, \
    convert_squeeze, convert_unsqueeze, convert_shape
from .elementwise_layers import convert_elementwise_add, convert_elementwise_mul, \
    convert_elementwise_div, convert_elementwise_sub
from .activation_layers import convert_relu, convert_lrelu, convert_selu, \
    convert_softmax, convert_sigmoid, convert_tanh, convert_hardtanh
from .pooling_layers import convert_avgpool, convert_maxpool, convert_maxpool3, \
    convert_adaptive_avg_pool2d, convert_adaptive_max_pool2d
from .normalization_layers import convert_batchnorm, convert_instancenorm, convert_dropout
from .linear_layers import convert_gemm, convert_matmul
from .embedding_layers import convert_gather
from .upsampling_layers import convert_upsample_bilinear, convert_upsample
from .padding_layers import convert_padding
from .operation_layers import convert_concat, convert_slice, convert_sum, \
    convert_reduce_sum, convert_slice, convert_clip
from .constant_layers import convert_constant


AVAILABLE_CONVERTERS = {
    'onnx::Conv': convert_conv,
    'onnx::ConvTranspose': convert_convtranspose,
    'onnx::Flatten': convert_flatten,
    'onnx::Gemm': convert_gemm,
    'onnx::MaxPool': convert_maxpool,
    'max_pool2d': convert_maxpool,
    'aten::max_pool3d': convert_maxpool3,
    'aten::max_pool2d_with_indices': convert_maxpool,
    'aten::max_pool2d': convert_maxpool,
    'aten::avg_pool2d': convert_avgpool,
    'onnx::AveragePool': convert_avgpool,
    'onnx::Dropout': convert_dropout,
    'onnx::BatchNormalization': convert_batchnorm,
    'onnx::InstanceNormalization': convert_instancenorm,
    'onnx::Add': convert_elementwise_add,
    'onnx::Mul': convert_elementwise_mul,
    'onnx::Div': convert_elementwise_div,
    'onnx::Sub': convert_elementwise_sub,
    'onnx::Sum': convert_sum,
    'onnx::Concat': convert_concat,
    'onnx::Relu': convert_relu,
    'onnx::LeakyRelu': convert_lrelu,
    'onnx::Sigmoid': convert_sigmoid,
    'onnx::Softmax': convert_softmax,
    'aten::softmax': convert_softmax,
    'onnx::Tanh': convert_tanh,
    'aten::hardtanh': convert_hardtanh,
    'onnx::Selu': convert_selu,
    'onnx::Transpose': convert_transpose,
    'onnx::Reshape': convert_reshape,
    'onnx::MatMul': convert_matmul,
    'onnx::Gather': convert_gather,
    'onnx::ReduceSum': convert_reduce_sum,
    'onnx::Constant': convert_constant,
    'aten::upsample_bilinear2d': convert_upsample_bilinear,
    'onnx::Upsample': convert_upsample,
    'onnx::Pad': convert_padding,
    'onnx::GlobalAveragePool': convert_adaptive_avg_pool2d,
    'aten::adaptive_avg_pool2d': convert_adaptive_avg_pool2d,
    'onnx::GlobalMaxPool': convert_adaptive_max_pool2d,
    'aten::adaptive_max_pool2d': convert_adaptive_max_pool2d,
    'onnx::Slice': convert_slice,
    'onnx::Squeeze': convert_squeeze,
    'onnx::Unsqueeze': convert_unsqueeze,
    'onnx::Shape': convert_shape,
    'onnx::Clip': convert_clip,
}
