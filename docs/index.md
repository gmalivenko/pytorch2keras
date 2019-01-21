# pytorch2keras

[![Build Status](https://travis-ci.com/nerox8664/pytorch2keras.svg?branch=master)](https://travis-ci.com/nerox8664/pytorch2keras)
[![GitHub License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-2.7%2C3.6-lightgrey.svg)](https://github.com/nerox8664/pytorch2keras)
[![Downloads](https://pepy.tech/badge/pytorch2keras)](https://pepy.tech/project/pytorch2keras)
![PyPI](https://img.shields.io/pypi/v/pytorch2keras.svg)

PyTorch to Keras model converter. This project is created to make a model conversation easier, so, the converter API is developed with maximal simplicity.


## PyTorch 0.4.1 and greater

There is [the problem related to a new version](https://github.com/pytorch/pytorch/issues/13963):

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
