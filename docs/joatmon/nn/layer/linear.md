#


## Linear
```python 
Linear(
   in_features: int, out_features: int, bias: bool = True
)
```


---
Applies a linear transformation to the incoming data: y = xA^T + b

# Arguments
in_features (int): size of each input sample
out_features (int): size of each output sample
bias (bool, optional): If set to False, the layer will not learn an additive bias. Default: True

---
# Attributes
    bias (Tensor):   the learnable bias of the module of shape (out_features)


**Methods:**


### .reset_parameters
```python
.reset_parameters()
```

---
Resets the parameters (weight, bias) to their initial values.

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
    Tensor: The output tensor after applying linear transformation.

### .extra_repr
```python
.extra_repr()
```

---
Returns a string containing a brief description of the module.

# Returns
str: A string containing a brief description of the module.
