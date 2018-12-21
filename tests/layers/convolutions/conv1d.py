import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


class LayerTest(nn.Module):
    def __init__(self, inp, out, kernel_size=3, padding=1, stride=1, bias=False, dilation=1):
        super(LayerTest, self).__init__()
        self.conv = nn.Conv1d(inp, out, kernel_size=kernel_size, padding=padding, \
            stride=stride, bias=bias, dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        return x


def check_error(output, k_model, input_np, epsilon=1e-5):
    pytorch_output = output.data.numpy()
    keras_output = k_model.predict(input_np)

    error = np.max(pytorch_output - keras_output)
    print('Error:', error)

    assert error < epsilon
    return error


if __name__ == '__main__':
    max_error = 0
    for kernel_size in [1, 3, 5]:
        for padding in [0, 1, 3]:
            for stride in [1, 2]:
                for bias in [True, False]:
                    for dilation in [1, 2, 3]:
                        # ValueError: strides > 1 not supported in conjunction with dilation_rate > 1
                        if stride > 1 and dilation > 1:
                            continue

                        ins = np.random.choice([1, 3, 7])
                        model = LayerTest(ins, np.random.choice([1, 3, 7]), \
                            kernel_size=kernel_size, padding=padding, stride=stride, bias=bias, dilation=dilation)
                        model.eval()

                        input_np = np.random.uniform(0, 1, (1, ins, 224))
                        input_var = Variable(torch.FloatTensor(input_np))
                        output = model(input_var)
                        print(output.size())
                        k_model = pytorch_to_keras(model, input_var, (ins, 224,), verbose=True)

                        error = check_error(output, k_model, input_np)
                        if max_error < error:
                            max_error = error

    print('Max error: {0}'.format(max_error))
