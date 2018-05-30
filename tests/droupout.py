import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


class TestDropout(nn.Module):
    """Module for Dropout conversion testing
    """

    def __init__(self, inp=10, out=16, p=0.5, bias=True):
        super(TestDropout, self).__init__()
        self.linear = nn.Linear(inp, out, bias=bias)
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        return x


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        inp = np.random.randint(1, 100)
        out = np.random.randint(1, 100)
        p = np.random.uniform(0, 1)
        model = TestDropout(inp, out, inp % 2, p)
        model.eval()

        input_np = np.random.uniform(-1.0, 1.0, (1, inp))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (inp,), verbose=True)

        keras_output = k_model.predict(input_np)

        pytorch_output = output.data.numpy()

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

        # not implemented yet
