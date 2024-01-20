#


## BatchNorm
```python 
BatchNorm(
   features, eps: float = 1e-05, momentum: float = 0.1, affine: bool = True,
   track_running_stats: bool = True
)
```


---
Applies Batch Normalization over a mini-batch of inputs.

# Arguments
features (int): Number of features in the input.
eps (float, optional): A value added to the denominator for numerical stability. Default: 1e-5
momentum (float, optional): The value used for the running_mean and running_var computation. Default: 0.1
affine (bool, optional): A boolean value that when set to True, gives the layer learnable affine parameters. Default: True
track_running_stats (bool, optional): A boolean value that when set to True, this module tracks the running mean and variance, and when set to False, this module does not track such statistics and always uses batch statistics in both training and eval modes. Default: True

---
# Attributes
    num_batches_tracked (int): The number of batches tracked. Default: None


**Methods:**


### .reset_running_stats
```python
.reset_running_stats()
```

---
Resets the running stats (running_mean, running_var, num_batches_tracked) to their initial values.

### .reset_parameters
```python
.reset_parameters()
```

---
Resets the parameters (weight, bias) to their initial values and resets the running stats.

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
    Tensor: The output tensor after applying batch normalization.
