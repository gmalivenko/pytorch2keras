# pytorch2keras

[![Build Status](https://travis-ci.com/gmalivenko/pytorch2keras.svg?branch=master)](https://travis-ci.com/gmalivenko/pytorch2keras)
[![GitHub License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-2.7%2C3.6-lightgrey.svg)](https://github.com/gmalivenko/pytorch2keras)
[![Downloads](https://pepy.tech/badge/pytorch2keras)](https://pepy.tech/project/pytorch2keras)
![PyPI](https://img.shields.io/pypi/v/pytorch2keras.svg)
[![Readthedocs](https://img.shields.io/readthedocs/pytorch2keras.svg)](https://pytorch2keras.readthedocs.io/en/latest/)


PyTorch to Keras model converter.

## Installation

```
pip install pytorch2keras 
```

## Important notice

To use the converter properly, please, make changes in your `~/.keras/keras.json`:


```json
...
"backend": "tensorflow",
"image_data_format": "channels_first",
...
```

## Tensorflow.js

For the proper conversion to a tensorflow.js format, please use the new flag `names='short'`.

Here is a short instruction how to get a tensorflow.js model:

1. First of all, you have to convert your model to Keras with this converter:

```python
k_model = pytorch_to_keras(model, input_var, [(10, 32, 32,)], verbose=True, names='short')  
```

2. Now you have Keras model. You can save it as h5 file and then convert it with `tensorflowjs_converter` but it doesn't work sometimes. As alternative, you may get Tensorflow Graph and save it as a frozen model:

```python
# Function below copied from here:
# https://stackoverflow.com/questions/45466020/how-to-export-keras-h5-to-tensorflow-pb 
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = \
            list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


from keras import backend as K
import tensorflow as tf
frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in k_model.outputs])

tf.train.write_graph(frozen_graph, ".", "my_model.pb", as_text=False)
print([i for i in k_model.outputs])

```

3. You will see the output layer name, so, now it's time to convert `my_model.pb` to tfjs model:

```bash
tensorflowjs_converter  \
    --input_format=tf_frozen_model \
    --output_node_names='TANHTObs/Tanh' \
    my_model.pb \
    model_tfjs
```

4. Thats all!

```js
const MODEL_URL = `model_tfjs/tensorflowjs_model.pb`;
const WEIGHTS_URL = `model_tfjs/weights_manifest.json`;
const model = await tf.loadFrozenModel(MODEL_URL, WEIGHTS_URL);
```

## How to use

It's the converter of PyTorch graph to a Keras (Tensorflow backend) model.

Firstly, we need to load (or create) a valid PyTorch model:

```python
class TestConv2d(nn.Module):
    """
    Module for Conv2d testing
    """

    def __init__(self, inp=10, out=16, kernel_size=3):
        super(TestConv2d, self).__init__()
        self.conv2d = nn.Conv2d(inp, out, stride=1, kernel_size=kernel_size, bias=True)

    def forward(self, x):
        x = self.conv2d(x)
        return x

model = TestConv2d()

# load weights here
# model.load_state_dict(torch.load(path_to_weights.pth))
```

The next step - create a dummy variable with correct shape:

```python
input_np = np.random.uniform(0, 1, (1, 10, 32, 32))
input_var = Variable(torch.FloatTensor(input_np))
```

We use the dummy-variable to trace the model (with jit.trace):

```python
from pytorch2keras import pytorch_to_keras
# we should specify shape of the input tensor
k_model = pytorch_to_keras(model, input_var, [(10, 32, 32,)], verbose=True)  
```

You can also set H and W dimensions to None to make your model shape-agnostic (e.g. fully convolutional netowrk):

```python
from pytorch2keras.converter import pytorch_to_keras
# we should specify shape of the input tensor
k_model = pytorch_to_keras(model, input_var, [(10, None, None,)], verbose=True)  
```

That's all! If all the modules have converted properly, the Keras model will be stored in the `k_model` variable.


## API

Here is the only method `pytorch_to_keras` from `pytorch2keras` module.

```python
def pytorch_to_keras(
    model, args, input_shapes=None,
    change_ordering=False, verbose=False, name_policy=None,
):
```

Options:

* `model` - a PyTorch model (nn.Module) to convert;
* `args` - a list of dummy variables with proper shapes;
* `input_shapes` - (experimental) list with overrided shapes for inputs;
* `change_ordering` - (experimental) boolean, if enabled, the converter will try to change `BCHW` to `BHWC`
* `verbose` - boolean, detailed log of conversion
* `name_policy` - (experimental) choice from [`keep`, `short`, `random`]. The selector set the target layer naming policy.

## Supported layers

* Activations:
    + ReLU
    + LeakyReLU
    + SELU
    + Sigmoid
    + Softmax
    + Tanh

* Constants

* Convolutions:
    + Conv2d
    + ConvTrsnpose2d

* Element-wise:
    + Add
    + Mul
    + Sub
    + Div

* Linear

* Normalizations:
    + BatchNorm2d
    + InstanceNorm2d

* Poolings:
    + MaxPool2d
    + AvgPool2d
    + Global MaxPool2d (adaptive pooling to shape [1, 1])


## Models converted with pytorch2keras

* ResNet*
* VGG*
* PreResNet*
* DenseNet*
* AlexNet
* Mobilenet v2

## Usage
Look at the `tests` directory.

## License
This software is covered by MIT License.
