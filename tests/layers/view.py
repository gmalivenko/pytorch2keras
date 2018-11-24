import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


class TestView(nn.Module):
    def __init__(self):
        super(TestView, self).__init__()
        self.conv2d = nn.Conv2d(22, 32, kernel_size=1, bias=True)
        self.fc = nn.Linear(15488, 3)

    def forward(self, x):
        x = self.conv2d(x)

        print(type(x.size()[0]))

        x = x.view([int(x.size(0)), -1])
        x = self.fc(x)
        return x


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        kernel_size = 1
        inp = 22
        out = 32

        model = TestView()

        input_np = np.random.uniform(0, 1, (1, inp, inp, inp))
        input_var = Variable(torch.FloatTensor(input_np))

        k_model = pytorch_to_keras(model, input_var, (inp, inp, inp,), verbose=True)

        output = model(input_var)
        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np)

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))
