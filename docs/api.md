Since version `0.2.1` the converter has the following API:

```
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