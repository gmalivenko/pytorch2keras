import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


class LayerTest(nn.Module):
    def __init__(self, inp, out, bias=False):
        super(LayerTest, self).__init__()
        self.fc = nn.Linear(inp, out, bias=bias)

    def forward(self, x):
        x = self.fc(x)
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
    for bias in [True, False]:
        ins = np.random.choice([1, 3, 7])
        model = LayerTest(ins, np.random.choice([1, 3, 7]), bias=bias)
        model.eval()

        input_np = np.random.uniform(0, 1, (1, ins))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)
        print(output.size())
        k_model = pytorch_to_keras(model, input_var, (ins,), verbose=True)

        error = check_error(output, k_model, input_np)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))
