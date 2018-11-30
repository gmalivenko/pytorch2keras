# pytorch2keras

[![Build Status](https://travis-ci.com/nerox8664/pytorch2keras.svg?branch=master)](https://travis-ci.com/nerox8664/pytorch2keras)
[![GitHub License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-2.7%2C3.6-lightgrey.svg)](https://github.com/nerox8664/pytorch2keras)
[![Downloads](https://pepy.tech/badge/pytorch2keras)](https://pepy.tech/project/pytorch2keras)
![PyPI](https://img.shields.io/pypi/v/pytorch2keras.svg)

PyTorch to Keras model convertor. 

## Installation

```
pip install pytorch2keras 
```

## Important notice

To use the converter properly, please, make changes in your `~/.keras/keras.json`:


```
...
"backend": "tensorflow",
"image_data_format": "channels_first",
...
```

## PyTorch 0.4.1 and greater

There are [some problem related to a new version](https://github.com/pytorch/pytorch/issues/13963):

To make it work, please, cast all your `.view()` parameters to `int`. For example:

```
class ResNet(torchvision.models.resnet.ResNet):
    def __init__(self, *args, **kwargs):
        super(ResNet, self).__init__(*args, **kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(int(x.size(0)), -1)  #  << Here
        x = self.fc(x)
        return x
```

## Tensorflow.js

For the proper convertion to the tensorflow.js format, please use a new flag `names='short'`.


## How to build the latest PyTorch

Please, follow [this guide](https://github.com/pytorch/pytorch#from-source) to compile the latest version.

Additional information for Arch Linux users:

* the latest gcc8 is incompatible with actual nvcc version
* the legacy gcc54 can't compile C/C++ modules because of compiler flags

## How to use

It's the convertor of pytorch graph to a Keras (Tensorflow backend) graph.

Firstly, we need to load (or create) pytorch model:

```
class TestConv2d(nn.Module):
    """Module for Conv2d convertion testing
    """

    def __init__(self, inp=10, out=16, kernel_size=3):
        super(TestConv2d, self).__init__()
        self.conv2d = nn.Conv2d(inp, out, stride=(inp % 3 + 1), kernel_size=kernel_size, bias=True)

    def forward(self, x):
        x = self.conv2d(x)
        return x

model = TestConv2d()

# load weights here
# model.load_state_dict(torch.load(path_to_weights.pth))
```

The next step - create a dummy variable with correct shapes:

```
input_np = np.random.uniform(0, 1, (1, 10, 32, 32))
input_var = Variable(torch.FloatTensor(input_np))
```

We're using dummy-variable in order to trace the model.

```
from converter import pytorch_to_keras
# we should specify shape of the input tensor
k_model = pytorch_to_keras(model, input_var, [(10, 32, 32,)], verbose=True)  
```

You can also set H and W dimensions to None to make your model shape-agnostic:

```
from converter import pytorch_to_keras
# we should specify shape of the input tensor
k_model = pytorch_to_keras(model, input_var, [(10, None, None,)], verbose=True)  
```

That's all! If all is ok, the Keras model is stores into the `k_model` variable.

## Supported layers

Layers:

* Linear
* Conv2d (also with groups)
* DepthwiseConv2d (with limited parameters)
* Conv3d
* ConvTranspose2d
* MaxPool2d
* MaxPool3d
* AvgPool2d
* Global average pooling (as special case of AdaptiveAvgPool2d)
* Embedding
* UpsamplingNearest2d
* BatchNorm2d
* InstanceNorm2d

Reshape:

* View
* Reshape
* Transpose

Activations:

* ReLU
* LeakyReLU
* Tanh
* HardTanh (clamp)
* Softmax
* Sigmoid

Element-wise:

* Addition
* Multiplication
* Subtraction

Misc:

* reduce sum ( .sum() method)

## Unsupported parameters

* Pooling: count_include_pad, dilation, ceil_mode

## Models converted with pytorch2keras

* ResNet*
* PreResNet*
* SqueezeNet (with ceil_mode=False)
* SqueezeNext
* DenseNet*
* AlexNet
* Inception
* SeNet
* Mobilenet v2

## Usage
Look at the `tests` directory.

## License
This software is covered by MIT License.