import unittest
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


class TestConvTranspose2d(nn.Module):
    """Module for ConvTranspose2d conversion testing
    """

    def __init__(self, inp=10, out=16, kernel_size=3, stride=1, bias=True, padding=0):
        super(TestConvTranspose2d, self).__init__()
        self.conv2d = nn.ConvTranspose2d(inp, out, kernel_size=kernel_size, bias=bias, stride=stride, padding=padding)

    def forward(self, x):
        x = self.conv2d(x)
        return x


class ConvTranspose2dTest(unittest.TestCase):
    N = 100

    def test(self):
        max_error = 0
        for i in range(self.N):
            kernel_size = np.random.randint(1, 7)
            inp = np.random.randint(kernel_size + 1, 100)
            out = np.random.randint(1, 100)

            model = TestConvTranspose2d(inp, out, kernel_size, 2, inp % 3)

            input_np = np.random.uniform(0, 1, (1, inp, inp, inp))
            input_var = Variable(torch.FloatTensor(input_np))
            output = model(input_var)

            k_model = pytorch_to_keras(model, input_var, (inp, inp, inp,), verbose=True)

            pytorch_output = output.data.numpy()
            keras_output = k_model.predict(input_np)

            error = np.max(pytorch_output - keras_output)
            print(error)
            if max_error < error:
                max_error = error

        print('Max error: {0}'.format(max_error))

    def test_with_padding(self):
        max_error = 0
        for i in range(self.N):
            kernel_size = np.random.randint(1, 7)
            inp = np.random.randint(kernel_size + 1, 100)
            out = np.random.randint(1, 100)

            model = TestConvTranspose2d(inp, out, kernel_size, 2, inp % 3, padding=1)

            input_np = np.random.uniform(0, 1, (1, inp, inp, inp))
            input_var = Variable(torch.FloatTensor(input_np))
            output = model(input_var)

            k_model = pytorch_to_keras(model, input_var, (inp, inp, inp,), verbose=True)

            pytorch_output = output.data.numpy()
            keras_output = k_model.predict(input_np)

            error = np.max(pytorch_output - keras_output)
            print(error)
            if max_error < error:
                max_error = error

        print('Max error: {0}'.format(max_error))


if __name__ == '__main__':
    unittest.main()
