#


## ExponentialLR
```python 
ExponentialLR(
   optimizer, gamma, last_epoch = -1, verbose = False
)
```


---
Implements the Exponential Learning Rate Scheduler.

This scheduler decays the learning rate of each parameter group by a specified gamma every epoch.

# Attributes
optimizer (Optimizer): The optimizer for which the learning rate will be scheduled.
gamma (float): The multiplicative factor of learning rate decay.
last_epoch (int, optional): The index of the last epoch. Default is -1.
verbose (bool, optional): If True, prints a message to stdout for each update. Default is False.


**Methods:**


### .get_lr
```python
.get_lr()
```

---
Computes the learning rate for the current epoch.

# Returns
list: The learning rates for each parameter group.
