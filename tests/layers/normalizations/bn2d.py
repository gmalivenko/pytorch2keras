import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras
import random


class LayerTest(nn.Module):
    def __init__(self, out, eps, momentum):
        super(LayerTest, self).__init__()
        self.bn = nn.BatchNorm2d(out, eps=eps, momentum=momentum)

    def forward(self, x):
        x = self.bn(x)
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
    for i in range(10):
        inp_size = np.random.randint(10, 100)

        model = LayerTest(inp_size, random.random(), random.random())
        model.eval()

        input_np = np.random.uniform(0, 1, (1, inp_size, 224, 224))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (inp_size, 224, 224,), verbose=True)

        error = check_error(output, k_model, input_np)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))
