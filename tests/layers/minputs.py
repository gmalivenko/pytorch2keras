import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


class TestMultipleInputs(nn.Module):
    """Module for multiple inputs conversion testing
    """

    def __init__(self, inp=10, out=16, kernel_size=3, bias=True):
        super(TestMultipleInputs, self).__init__()
        self.conv2d = nn.Conv2d(inp, out, kernel_size=kernel_size, bias=bias)
        self.deconv2d = nn.ConvTranspose2d(inp, out, kernel_size=kernel_size, bias=bias)
        self.in2d = nn.InstanceNorm2d(out)

    def forward(self, x, y, z):
        return self.in2d(self.deconv2d(x)) + self.in2d(self.deconv2d(y)) + self.in2d(self.deconv2d(z))


def check_error(output, k_model, input_np, epsilon=1e-5):
    pytorch_output = output.data.numpy()
    keras_output = k_model.predict([input_np, input_np, input_np])

    error = np.max(pytorch_output - keras_output)
    print('Error:', error)

    assert error < epsilon
    return error


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        kernel_size = np.random.randint(1, 7)
        inp = np.random.randint(kernel_size + 1, 100)
        out = np.random.randint(1, 100)

        model = TestMultipleInputs(inp, out, kernel_size, inp % 2)

        input_np = np.random.uniform(0, 1, (1, inp, inp, inp))
        input_var = Variable(torch.FloatTensor(input_np))
        input_var2 = Variable(torch.FloatTensor(input_np))
        input_var3 = Variable(torch.FloatTensor(input_np))

        output = model(input_var, input_var2, input_var3)

        k_model = pytorch_to_keras(
            model,
            [input_var, input_var2, input_var3],
            [(inp, inp, inp,), (inp, inp, inp,), (inp, inp, inp,)],
            verbose=True
        )

        error = check_error(output, k_model, input_np)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))
