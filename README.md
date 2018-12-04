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
* VGG*
* PreResNet*
* SqueezeNet (with ceil_mode=False)
* SqueezeNext
* DenseNet*
* AlexNet
* Inception
* SeNet
* Mobilenet v2
* DiracNet
* DARTS
* DRNC

| Model | Top1 | Top5 | Params | FLOPs | Source weights | Remarks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet-10 | 37.09 | 15.55 | 5,418,792 | 892.62M | osmr's repo | Success |
| ResNet-12 | 35.86 | 14.46 | 5,492,776 | 1,124.23M | osmr's repo | Success |
| ResNet-14 | 32.85 | 12.41 | 5,788,200 | 1,355.64M | osmr's repo | Success |
| ResNet-16 | 30.68 | 11.10 | 6,968,872 | 1,586.95M | osmr's repo | Success |
| ResNet-18 x0.25 | 49.16 | 24.45 | 831,096 | 136.64M | osmr's repo | Success |
| ResNet-18 x0.5 | 36.54 | 14.96 | 3,055,880 | 485.22M | osmr's repo | Success |
| ResNet-18 x0.75 | 33.25 | 12.54 | 6,675,352 | 1,045.75M | osmr's repo | Success |
| ResNet-18 | 29.13 | 9.94 | 11,689,512 | 1,818.21M | osmr's repo | Success |
| ResNet-34 | 25.34 | 7.92 | 21,797,672 | 3,669.16M | osmr's repo | Success |
| ResNet-50 | 23.50 | 6.87 | 25,557,032 | 3,868.96M | osmr's repo | Success |
| ResNet-50b | 22.92 | 6.44 | 25,557,032 | 4,100.70M | osmr's repo | Success |
| ResNet-101 | 21.66 | 5.99 | 44,549,160 | 7,586.30M | osmr's repo | Success |
| ResNet-101b | 21.18 | 5.60 | 44,549,160 | 7,818.04M | osmr's repo | Success |
| ResNet-152 | 21.01 | 5.61 | 60,192,808 | 11,304.85M | osmr's repo | Success |
| ResNet-152b | 20.54 | 5.37 | 60,192,808 | 11,536.58M | osmr's repo | Success |
| PreResNet-18 | 28.72 | 9.88 | 11,687,848 | 1,818.41M | osmr's repo | Success |
| PreResNet-34 | 25.88 | 8.11 | 21,796,008 | 3,669.36M | osmr's repo | Success |
| PreResNet-50 | 23.39 | 6.68 | 25,549,480 | 3,869.16M | osmr's repo | Success |
| PreResNet-50b | 23.16 | 6.64 | 25,549,480 | 4,100.90M | osmr's repo | Success |
| PreResNet-101 | 21.45 | 5.75 | 44,541,608 | 7,586.50M | osmr's repo | Success |
| PreResNet-101b | 21.73 | 5.88 | 44,541,608 | 7,818.24M | osmr's repo | Success |
| PreResNet-152 | 20.70 | 5.32 | 60,185,256 | 11,305.05M | osmr's repo | Success |
| PreResNet-152b | 21.00 | 5.75 | 60,185,256 | 11,536.78M | Gluon Model Zoo| Success |
| PreResNet-200b | 21.10 | 5.64 | 64,666,280 | 15,040.27M | tornadomeet/ResNet | Success |
| DenseNet-121 | 25.11 | 7.80 | 7,978,856 | 2,852.39M | Gluon Model Zoo| Success |
| DenseNet-161 | 22.40 | 6.18 | 28,681,000 | 7,761.25M | Gluon Model Zoo| Success |
| DenseNet-169 | 23.89 | 6.89 | 14,149,480 | 3,381.48M | Gluon Model Zoo| Success |
| DenseNet-201 | 22.71 | 6.36 | 20,013,928 | 4,318.75M | Gluon Model Zoo| Success |
<!-- | DPN-68 | 23.57 | 7.00 | 12,611,602 | 2,338.71M | Cadene's repo | Success |
| DPN-98 | 20.23 | 5.28 | 61,570,728 | 11,702.80M | Cadene's repo | Success |
| DPN-131 | 20.03 | 5.22 | 79,254,504 | 16,056.22M | Cadene's repo | Success | -->
| DarkNet Tiny | 40.31 | 17.46 | 1,042,104 | 496.34M | osmr's repo | Success |
| DarkNet Ref | 38.00 | 16.68 | 7,319,416 | 365.55M | osmr's repo | Success |
| SqueezeNet v1.0 | 40.97 | 18.96 | 1,248,424 | 828.30M | osmr's repo | Success |
| SqueezeNet v1.1 | 39.09 | 17.39 | 1,235,496 | 354.88M | osmr's repo | Success |
| MobileNet x0.25 | 45.78 | 22.18 | 470,072 | 42.30M | osmr's repo | Success |
| MobileNet x0.5 | 36.12 | 14.81 | 1,331,592 | 152.04M | osmr's repo | Success |
| MobileNet x0.75 | 32.71 | 12.28 | 2,585,560 | 329.22M | Gluon Model Zoo| Success |
| MobileNet x1.0 | 29.25 | 10.03 | 4,231,976 | 573.83M | Gluon Model Zoo| Success |
| FD-MobileNet x0.25 | 56.19 | 31.38 | 383,160 | 12.44M | osmr's repo | Success |
| FD-MobileNet x0.5 | 42.62 | 19.69 | 993,928 | 40.93M | osmr's repo | Success |
| FD-MobileNet x1.0 | 35.95 | 14.72 | 2,901,288 | 146.08M | clavichord93/FD-MobileNet | Success |
| MobileNetV2 x0.25 | 48.89 | 25.24 | 1,516,392 | 32.22M | Gluon Model Zoo| Success |
| MobileNetV2 x0.5 | 35.51 | 14.64 | 1,964,736 | 95.62M | Gluon Model Zoo| Success |
| MobileNetV2 x0.75 | 30.82 | 11.26 | 2,627,592 | 191.61M | Gluon Model Zoo| Success |
| MobileNetV2 x1.0 | 28.51 | 9.90 | 3,504,960 | 320.19M | Gluon Model Zoo| Success |
<!-- | NASNet-A-Mobile | 25.37 | 7.95 | 5,289,978 | 587.29M | Cadene's repo | Success | -->
| InceptionV3 | 21.22 | 5.59 | 23,834,568 | 5,746.72M | Gluon Model Zoo| Success |
<!-- | AirNet50-1x64d (r=2) | 22.48 | 6.21 | 27,425,864 | 4,757.77M | soeaver/AirNet-PyTorch | Success |
| AirNet50-1x64d (r=16) | 22.91 | 6.46 | 25,714,952 | 4,385.54M | soeaver/AirNet-PyTorch | Success | -->
<!-- | AirNeXt50-32x4d (r=2) | 20.87 | 5.51 | 27,604,296 | 5,321.18M | soeaver/AirNet-PyTorch | Success | -->
| DiracNetV2-18 | 31.47 | 11.70 | 11,511,784 | 1,798.43M | szagoruyko/diracnets | Success |
| DiracNetV2-34 | 28.75 | 9.93 | 21,616,232 | 3,649.37M | szagoruyko/diracnets | Success |
| DARTS | 26.70 | 8.74 | 4,718,752 | 537.64M | szagoruyko/diracnets | Success |

## Usage
Look at the `tests` directory.

## License
This software is covered by MIT License.