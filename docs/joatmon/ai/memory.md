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
Adds an experience to the memory buffer.


**Args**

* **element** (tuple) : A tuple representing an experience. It includes the state, action, reward, next_state, and terminal flag.


### .sample
```python
.sample()
```

---
Samples a batch of experiences from the memory buffer.


**Returns**

* **list**  : A list of experiences.


----


## RingMemory
```python 
RingMemory(
   batch_size = 32, size = 960000
)
```


---
Ring Memory

This class is used to create a ring buffer memory for storing and sampling experiences in reinforcement learning.
It inherits from the CoreMemory class and overrides its buffer with a RingBuffer.


**Args**

* **batch_size** (int) : The size of the batch to be sampled from the buffer.
* **size** (int) : The maximum size of the ring buffer.

