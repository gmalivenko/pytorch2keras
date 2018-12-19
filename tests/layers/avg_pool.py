import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


class AvgPool(nn.Module):
    """Module for AveragePool conversion testing
    """

    def __init__(self, stride=3, padding=0, kernel_size=3):
        super(AvgPool, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, padding=padding, stride=stride)

    def forward(self, x):
        x = self.pool(x)
        return x


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        kernel_size = np.random.randint(2, 7)
        stride = np.random.randint(1, kernel_size)
        padding = np.random.randint(1, kernel_size/2 + 1)
        inp = np.random.randint(kernel_size + 1, 100)

        model = AvgPool(kernel_size=kernel_size, padding=padding, stride=stride)

        input_np = np.random.uniform(0, 1, (1, inp, inp, inp))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (inp, inp, inp,), verbose=False, names='keep')
        print(k_model.summary())
        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np)

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))
