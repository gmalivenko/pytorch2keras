import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


class TestSum(nn.Module):
    def __init__(self, input_size):
        super(TestSum, self).__init__()
        self.embedd = nn.Embedding(input_size, 100)

    def forward(self, input):
        return self.embedd(input).sum(dim=0)


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        input_np = np.random.randint(0, 10, (1, 1, 4))
        input = Variable(torch.LongTensor(input_np))

        simple_net = TestSum(1000)
        output = simple_net(input)

        k_model = pytorch_to_keras(simple_net, input, (1, 4), verbose=True)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np)

        error = np.max(pytorch_output - keras_output[0])
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))
