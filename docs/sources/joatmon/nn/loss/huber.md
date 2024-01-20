#


## HuberLoss
```python 
HuberLoss(
   delta = 1.0
)
```


---
Implements the Huber loss function.

Huber loss is less sensitive to outliers in data than mean squared error.
It's quadratic for small values of the input and linear for large values.

# Attributes
delta (float): The point where the Huber loss function changes from a quadratic to linear.
_loss (np.array): The computed loss value.


**Methods:**


### .forward
```python
.forward(
   prediction, target
)
```

---
Initializes the HuberLoss class.

# Arguments
delta (float, optional): The point where the Huber loss function changes from a quadratic to linear. Default is 1.0.
