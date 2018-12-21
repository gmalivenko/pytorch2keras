import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


class LayerTest(nn.Module):
    def __init__(self,  kernel_size=3, padding=1, stride=1):
        super(LayerTest, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, padding=padding, stride=stride)

    def forward(self, x):
        x = self.pool(x)
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
    for kernel_size in [1, 3, 5, 7]:
        for padding in [0, 1, 3]:
            for stride in [1, 2, 3, 4]:
                # RuntimeError: invalid argument 2: pad should be smaller than half of kernel size, but got padW = 1, padH = 1, kW = 1,
                if padding > kernel_size / 2:
                    continue

                model = LayerTest(kernel_size=kernel_size, padding=padding, stride=stride)
                model.eval()

                input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
                input_var = Variable(torch.FloatTensor(input_np))
                output = model(input_var)

                k_model = pytorch_to_keras(model, input_var, (3, 224, 224,), verbose=True)

                error = check_error(output, k_model, input_np)
                if max_error < error:
                    max_error = error

    print('Max error: {0}'.format(max_error))
