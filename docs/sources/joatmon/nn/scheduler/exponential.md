#


## ExponentialLR
```python 
ExponentialLR(
   optimizer, gamma, last_epoch = -1, verbose = False
)
```


---
Decays the learning rate of each parameter group by gamma every epoch.
When last_epoch=-1, sets initial lr as lr.


**Args**

* **optimizer** (Optimizer) : Wrapped optimizer.
* **gamma** (float) : Multiplicative factor of learning rate decay.
* **last_epoch** (int) : The index of last epoch. Default: -1.
* **verbose** (bool) : If ``True``, prints a message to stdout for
    each update. Default: ``False``.



**Methods:**


### .get_lr
```python
.get_lr()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.
