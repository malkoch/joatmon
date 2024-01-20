#


## Array
```python 
Array(
   data = None, dtype = 'float32'
)
```


---
Array class for handling multi-dimensional data.

This class provides a way to handle multi-dimensional data in a flattened format,
while still providing access to the data in its original shape.


**Attributes**

* **_data** (list) : The flattened data.
* **_dtype** (str) : The data type of the elements in the array.
* **_shape** (tuple) : The shape of the original data.


**Args**

* **data** (list) : The original multi-dimensional data.
* **dtype** (str) : The data type of the elements in the array.



**Methods:**


### .astype
```python
.astype(
   dtype, copy = True
)
```

---
Convert the Array to a specified data type.


**Args**

* **dtype** (str) : The data type to convert to.
* **copy** (bool) : Whether to return a new Array or modify the existing one.


### .T
```python
.T()
```

---
Return the transpose of the Array.

### .ndim
```python
.ndim()
```

---
Return the number of dimensions of the Array.

### .shape
```python
.shape()
```

---
Return the shape of the Array.

### .dtype
```python
.dtype()
```

---
Return the data type of the elements in the Array.
