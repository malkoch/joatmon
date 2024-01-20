#


## Upsample
```python 
Upsample(
   scale_factor = None, mode = 'nearest'
)
```


---
Upsamples an input.

The input data is assumed to be of the form `minibatch x channels x [optional depth] x [optional height] x width`.
The modes available for upsampling are: `nearest`.

# Arguments
scale_factor (int or tuple, optional): multiplier for spatial size. Has to match input size if it is a tuple.
mode (str, optional): the upsampling algorithm: one of `nearest`. Default: `nearest`

---
# Attributes
    _mode (str): the upsampling algorithm: one of `nearest`.


**Methods:**


### .forward
```python
.forward(
   inp
)
```

---
Defines the computation performed at every call.

# Arguments
inp (Tensor): The input tensor.

---
# Returns
    Tensor: The output tensor after applying upsampling.
