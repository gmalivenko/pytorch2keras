import numpy as np
import torch
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras
import torchvision


if __name__ == '__main__':
    max_error = 0
    for i in range(10):
        model = torchvision.models.DenseNet()
        for m in model.modules():
            m.training = False

        input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (3, 224, 224,), verbose=True)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np)

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))
