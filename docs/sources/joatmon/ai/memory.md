#


## CoreMemory
```python 
CoreMemory(
   buffer, batch_size
)
```


---
Abstract base class for all implemented memory.

Do not use this abstract base class directly but instead use one of the concrete memory implemented.

To implement your own memory, you have to implement the following methods:

- `remember`
- `sample`


**Methods:**


### .remember
```python
.remember(
   element
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .sample
```python
.sample()
```

---
Sample an experience replay batch with size.

# Returns
batch (abstract): Randomly selected batch
from experience replay memory.

----


## RingMemory
```python 
RingMemory(
   batch_size = 32, size = 960000
)
```


---
Ring Memory

# Arguments
size (int): .
