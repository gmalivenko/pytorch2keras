## Supported layers

* Activations:
    + ReLU
    + LeakyReLU
    + SELU
    + Sigmoid
    + Softmax
    + Tanh
    + HardTanh

* Constants

* Convolutions:
    + Conv1d
    + Conv2d
    + ConvTrsnpose2d

* Element-wise:
    + Add
    + Mul
    + Sub
    + Div

* Embedding

* Linear

* Normalizations:
    + BatchNorm2d
    + InstanceNorm2d
    + Dropout

* Poolings:
    + MaxPool2d
    + AvgPool2d
    + Global MaxPool2d (adaptive pooling to shape [1, 1])
    + Global AvgPool2d (adaptive pooling to shape [1, 1])

* Not tested yet:
    + Upsampling
    + Padding
    + Reshape


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
| InceptionV3 | 21.22 | 5.59 | 23,834,568 | 5,746.72M | Gluon Model Zoo| Success |
| DiracNetV2-18 | 31.47 | 11.70 | 11,511,784 | 1,798.43M | szagoruyko/diracnets | Success |
| DiracNetV2-34 | 28.75 | 9.93 | 21,616,232 | 3,649.37M | szagoruyko/diracnets | Success |
| DARTS | 26.70 | 8.74 | 4,718,752 | 537.64M | szagoruyko/diracnets | Success |
