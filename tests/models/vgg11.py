import numpy as np
import torch
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras
import torchvision


class VGG(torchvision.models.vgg.VGG):
    def __init__(self, *args, **kwargs):
        super(VGG, self).__init__(*args, **kwargs)

    def forward(self, x):
        x = self.features(x)
        x = x.view([int(x.size(0)), -1])
        x = self.classifier(x)
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
    for i in range(100):
        model = VGG(torchvision.models.vgg.make_layers(torchvision.models.vgg.cfg['A'], batch_norm=True))
        model.eval()

        input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (3, 224, 224,), verbose=True)

        error = check_error(output, k_model, input_np)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))
