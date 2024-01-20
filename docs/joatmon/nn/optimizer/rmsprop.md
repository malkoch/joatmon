#


## RMSprop
```python 
RMSprop(
   params, lr = 0.01, alpha = 0.99, eps = 1e-08, weight_decay = 0, momentum = 0,
   centered = False
)
```


---
Implements the RMSprop optimization algorithm.

RMSprop is an optimization algorithm designed to speed up training in deep neural networks. It adjusts the Adagrad method in a very simple way in an attempt to reduce its aggressive, monotonically decreasing learning rate.

# Attributes
params (iterable): An iterable of parameters to optimize or dicts defining parameter groups.
lr (float, optional): The learning rate. Default is 1e-2.
alpha (float, optional): Smoothing constant. Default is 0.99.
eps (float, optional): Term added to the denominator to improve numerical stability. Default is 1e-8.
weight_decay (float, optional): Weight decay (L2 penalty). Default is 0.
momentum (float, optional): Momentum factor. Default is 0.
centered (bool, optional): If True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance. Default is False.


**Methods:**


### .step
```python
.step()
```

---
Performs a single optimization step.

This function is called once per optimization step to update the parameters.
