import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


def group_conv1x1(in_channels,
                  out_channels,
                  groups):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        padding=1,
        groups=groups,
        bias=False)


class TestGroupConv2d(nn.Module):
    """Module for Conv2d conversion testing
    """

    def __init__(self, inp=10, groups=1):
        super(TestGroupConv2d, self).__init__()
        self.conv2d_group = group_conv1x1(inp, inp, groups)

    def forward(self, x):
        x = self.conv2d_group(x)
        return x


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        kernel_size = np.random.randint(1, 7)
        groups = np.random.randint(1, 10)
        inp = np.random.randint(kernel_size + 1, 10) * groups
        h, w = 32, 32
        model = TestGroupConv2d(inp, groups)

        input_np = np.random.uniform(0, 1, (1, inp, h, w))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (inp, h, w,), verbose=True)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np)

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))
