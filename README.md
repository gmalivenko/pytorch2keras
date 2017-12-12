# pytorch2keras
Pytorch to Keras model convertor. Very beta for now.

## Important notice

In that moment the only PyTorch 0.2 is supported. You can follow #6 for the further information about 0.3.

To use converter properly, please, make changes in your `~/.keras/keras.json`:

```
...
"backend": "tensorflow",
"image_data_format": "channels_first",
...
```

Note 1: some layers parameters (like ceiling and etc) isn't supported.

Note 2: recurrent layers isn't supported too.

## How to

It's a convertor of pytorch graph to a keras (tensorflow backend) graph.

Firstly, we need to load (or create) pytorch model.

For example:

```
class TestConv2d(nn.Module):
    """Module for Conv2d conversion testing
    """

    def __init__(self, inp=10, out=16, kernel_size=3, bias=True):
        super(TestConv2d, self).__init__()
        self.conv2d = nn.Conv2d(inp, out, stride=(inp % 3 + 1), kernel_size=kernel_size, bias=bias)

    def forward(self, x):
        x = self.conv2d(x)
        return x

model = TestConv2d()
# load weights here
# model.load_state_dict(torch.load(path_to_weights.pth))
```

The next step - iterate model with some data (for gradients computing):

```
input_np = np.random.uniform(0, 1, (1, 10, 32, 32))
input_var = Variable(torch.FloatTensor(input_np))
output = model(input_var)
```

We've got some useless output. Now we use this output variable to traverse the graph.

```
from converter import pytorch_to_keras
k_model = pytorch_to_keras((10, 32, 32,), output)  #we should specify shape of the input tensor
```

That's all! If all is ok, the Keras model was stored to the `k_model` variable.

## Supported layers

Layers:

* Linear
* Conv2d
* ConvTranspose2d
* MaxPool2d
* AvgPool2d

Reshape:

* View

Activations:

* ReLU
* LeakyReLU
* PReLU
* SELU
* Tanh
* Softmax
* Softplus
* Softsign
* Sigmoid

Element-wise:

* Addition
* Multiplication
* Subtraction

## Unsupported parameters

* Pooling: count_include_pad, dilation, ceil_mode
* Convolution: group

## Models converted with pytorch2keras

* ResNet18
* ResNet50
* SqueezeNet
* DenseNet
* AlexNet
* Inception (v4 only)

## Usage
Look at the `tests` directory.

## License
This software is covered by MIT License.