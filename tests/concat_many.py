import keras  # work around segfault
import sys
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

sys.path.append('../pytorch2keras')
from converter import pytorch_to_keras


class TestConcatMany(nn.Module):
    """Module for Concatenation (2 or many layers) testing
    """

    def __init__(self, inp=10, out=16, kernel_size=3, bias=True):
        super(TestConcatMany, self).__init__()
        k = np.random.randint(2, 30)
        self.conv2ds = \
            [nn.Conv2d(inp, out, kernel_size=kernel_size, bias=bias) for i in range(k)]

    def forward(self, x):
        conv2ds_results = [c(x) for c in self.conv2ds]
        x = torch.cat(conv2ds_results, dim=1)
        return x


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        kernel_size = np.random.randint(1, 7)
        inp = np.random.randint(kernel_size + 1, 100)
        out = np.random.randint(1, 100)

        model = TestConcatMany(inp, out, kernel_size, inp % 2)

        input_np = np.random.uniform(0, 1, (1, inp, inp, inp))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras((inp, inp, inp,), output)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np)

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))
