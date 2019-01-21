## Linear layer problem with PyTorch 0.4.1 and greater

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

## Recurrent layers

The recurrent layers are not supported due to complicated onnx-translation. The support is planned, but haven't implemented yet.