import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


class TestConv3d(nn.Module):
    """Module for Conv2d conversion testing
    """

    def __init__(self, inp=10, out=16, kernel_size=3, bias=True):
        super(TestConv3d, self).__init__()
        self.conv3d = nn.Conv3d(inp, out, kernel_size=kernel_size, bias=bias)

    def forward(self, x):
        x = self.conv3d(x)
        return x


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        kernel_size = np.random.randint(1, 7)
        inp = np.random.randint(kernel_size + 1, 30)
        out = np.random.randint(1, 30)

        model = TestConv3d(inp, out, kernel_size, inp % 2)

        input_var = Variable(torch.randn(1, inp, inp, inp, inp))

        output = model(input_var)

        k_model = pytorch_to_keras(model,
                                   input_var,
                                   (inp, inp, inp, inp,),
                                   verbose=True)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_var.numpy())
        error = np.max(pytorch_output - keras_output)
        print("iteration: {}, error: {}".format(i, error))
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))
