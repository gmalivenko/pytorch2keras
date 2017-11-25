import keras  # work around segfault
import sys
import numpy as np
import math

import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable

sys.path.append('../pytorch2keras')
from converter import pytorch_to_keras

if __name__ == '__main__':
    max_error = 0
    for i in range(10):
        model = torchvision.models.SqueezeNet()
        for m in model.modules():
            m.training = False

        input_np = np.random.uniform(0, 1, (1, 3, 299, 299))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras((3, 299, 299,), output)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np)

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))
