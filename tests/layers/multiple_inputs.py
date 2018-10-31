import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


class TestMultipleInputs(nn.Module):
    """Module for Conv2d conversion testing
    """

    def __init__(self, inp=10, out=16, kernel_size=3, bias=True):
        super(TestMultipleInputs, self).__init__()
        self.conv2d = nn.Conv2d(inp, out, kernel_size=kernel_size, bias=bias)

    def forward(self, x, y, z):
        return self.conv2d(x) + self.conv2d(y) + self.conv2d(z)


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
        k_model.summary()
        pytorch_output = output.data.numpy()
        keras_output = k_model.predict([input_np, input_np, input_np])

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))
