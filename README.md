# pytorch2keras

[![Build Status](https://travis-ci.com/nerox8664/pytorch2keras.svg?branch=master)](https://travis-ci.com/nerox8664/pytorch2keras)

Pytorch to Keras model convertor. Still beta for now.

## Installation

```
pip install pytorch2keras 
```

## Important notice

In that moment the only PyTorch 0.2 (deprecated) and PyTorch 0.4.0 (latest stable) are supported.

To use the converter properly, please, make changes in your `~/.keras/keras.json`:


```
...
"backend": "tensorflow",
"image_data_format": "channels_first",
...
```

The latest version of PyTorch (0.4.1) isn't supported yet.


## Python 3.7

There are some problem related to a new version:

Q. PyTorch 0.4 hadn't released wheel package for Python 3.7

A. You can build it from source:

```
git clone https://github.com/pytorch/pytorch

cd pytorch

git checkout v0.4.0

NO_CUDA=1 python setup.py install
```

Q. Tensorflow isn't available for Python 3.7

A. Yes, we're waiting for it.


## Tensorflow.js

For the proper convertion to the tensorflow.js format, please use a new flag `short_names=True`.


## How to build the latest PyTorch

Please, follow [this guide](https://github.com/pytorch/pytorch#from-source) to compile the latest version.

Additional information for Arch Linux users:

* the latest gcc8 is incompatible with actual nvcc version
* the legacy gcc54 can't compile C/C++ modules because of compiler flags

## How to use

It's a convertor of pytorch graph to a Keras (Tensorflow backend) graph.

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

Reshape:

* View
* Reshape
* Transpose

Activations:

* ReLU
* LeakyReLU
* PReLU (only with 0.2)
* SELU (only with 0.2)
* Tanh
* HardTanh (clamp)
* Softmax
* Softplus (only with 0.2)
* Softsign (only with 0.2)
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