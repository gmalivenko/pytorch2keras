import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


def channel_shuffle(x, groups):
    """Channel Shuffle operation from ShuffleNet [arxiv: 1707.01083]
    Arguments:
        x (Tensor): tensor to shuffle.
        groups (int): groups to be split
    """
    batch, channels, height, width = x.size()
    channels_per_group = channels // groups
    x = x.view(batch, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch, channels, height, width)
    return x


class TestChannelShuffle2d(nn.Module):
    """Module for Channel shuffle conversion testing
    """

    def __init__(self, inp=10, out=16, groups=32):
        super(TestChannelShuffle2d, self).__init__()
        self.groups = groups
        self.conv2d = nn.Conv2d(inp, out, kernel_size=3, bias=False)

    def forward(self, x):
        x = self.conv2d(x)
        x = channel_shuffle(x, self.groups)
        return x


if __name__ == '__main__':
    max_error = 0
    for i in range(100):
        groups = np.random.randint(1, 32)
        inp = np.random.randint(3, 32)
        out = np.random.randint(3, 32) * groups

        model = TestChannelShuffle2d(inp, out, groups)

        input_np = np.random.uniform(0, 1, (1, inp, inp, inp))
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, (inp, inp, inp,), verbose=True)

        pytorch_output = output.data.numpy()
        keras_output = k_model.predict(input_np)

        error = np.max(pytorch_output - keras_output)
        print(error)
        if max_error < error:
            max_error = error

    print('Max error: {0}'.format(max_error))
