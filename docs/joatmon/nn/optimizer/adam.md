#


## Adam
```python 
Adam(
   params, lr = 0.001, betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0,
   amsgrad = False
)
```


---
Implements the Adam optimization algorithm.

Adam is an optimization algorithm that can be used instead of the classical stochastic gradient descent procedure
to update network weights iterative based on training data.

# Attributes
params (iterable): An iterable of parameters to optimize or dicts defining parameter groups.
lr (float, optional): The learning rate. Default is 1e-3.
betas (Tuple[float, float], optional): Coefficients used for computing running averages of gradient and its square. Default is (0.9, 0.999).
eps (float, optional): Term added to the denominator to improve numerical stability. Default is 1e-8.
weight_decay (float, optional): Weight decay (L2 penalty). Default is 0.
amsgrad (bool, optional): Whether to use the AMSGrad variant of this algorithm. Default is False.


**Methods:**


### .step
```python
.step()
```

---
Performs a single optimization step.

This function is called once per optimization step to update the parameters.
