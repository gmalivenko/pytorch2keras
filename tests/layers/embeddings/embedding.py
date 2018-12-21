import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


class LayerTest(nn.Module):
    def __init__(self, input_size, embedd_size):
        super(LayerTest, self).__init__()
        self.embedd = nn.Embedding(input_size, embedd_size)

    def forward(self, x):
        x = self.embedd(x)
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
        emb_size = np.random.randint(10, 1000)
        inp_size = np.random.randint(10, 1000)

        model = LayerTest(inp_size, emb_size)
        model.eval()

        input_np = np.random.uniform(0, 1, (1, 1, inp_size))
        input_var = Variable(torch.LongTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, [(1, inp_size)], verbose=True)

        error = check_error(output, k_model, input_np)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))
