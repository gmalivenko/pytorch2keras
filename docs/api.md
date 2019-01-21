Here is the only method `pytorch_to_keras` from `pytorch2keras` module.
```
def pytorch_to_keras(
    model, args, input_shapes,
    change_ordering=False, training=False, verbose=False, names=False,
)
```

Options:

* model -- a PyTorch module to convert;
* args -- list of dummy variables with proper shapes;
* input_shapes -- list with shape tuples;
* change_ordering -- boolean, if enabled, the converter will try to change `BCHW` to `BHWC`
* training -- boolean, switch model to training mode (never use it)
* verbose -- boolean, verbose output
* names -- choice from [`keep`, `short`, `random`]. The selector set the target layer naming policy.