#


## CoreBuffer
```python 
CoreBuffer(
   values, batch_size
)
```


---
CoreBuffer class that inherits from the list class. It provides the functionality for a buffer with core operations.


**Attributes**

* **values** (list) : The list of values in the buffer.
* **batch_size** (int) : The size of the batch for sampling.

---
Methods:
    sample: Returns a random sample from the buffer.


**Methods:**


### .add
```python
.add(
   element
)
```

---
Adds an element to the buffer.


**Args**

* **element** (any) : The element to be added to the buffer.


### .sample
```python
.sample()
```

---
Returns a random sample from the buffer.


**Returns**

* **list**  : A random sample from the buffer.


----


## RingBuffer
```python 
RingBuffer(
   size, batch_size
)
```


---
RingBuffer class that inherits from the CoreBuffer class. It provides the functionality for a circular buffer.


**Attributes**

* **data** (list) : The list of data in the buffer.
* **start** (int) : The start index of the buffer.
* **end** (int) : The end index of the buffer.

---
Methods:
    add: Adds an element to the buffer.


**Methods:**


### .add
```python
.add(
   element
)
```

---
Adds an element to the buffer.

If the buffer is full, it removes the oldest element to make room for the new element.


**Args**

* **element** (any) : The element to be added to the buffer.

