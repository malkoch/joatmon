#


### load
```python
.load(
   network, path
)
```

---
Load the weights of a network from a file.


**Args**

* **network** (nn.Module) : The PyTorch network for which the weights are loaded.
* **path** (str) : The path to the directory containing the weights file.


----


### save
```python
.save(
   network, path
)
```

---
Save the weights of a network to a file.


**Args**

* **network** (nn.Module) : The PyTorch network for which the weights are saved.
* **path** (str) : The path to the directory where the weights file will be saved.


----


### display
```python
.display(
   values, positions
)
```

---
Display a list of values in a formatted string.


**Args**

* **values** (list) : The list of values to be displayed.
* **positions** (list) : The list of positions where each value should be displayed in the string.


----


### easy_range
```python
.easy_range(
   begin = 0, end = None, increment = 1
)
```

---
Generate a range of numbers.


**Args**

* **begin** (int) : The number at which the range begins.
* **end** (int) : The number at which the range ends.
* **increment** (int) : The increment between each number in the range.


**Returns**

* **generator**  : A generator that yields the numbers in the range.


----


### normalize
```python
.normalize(
   array, minimum = 0.0, maximum = 255.0, dtype = 'float32'
)
```

---
Normalize an array to a specified range.


**Args**

* **array** (numpy array) : The array to be normalized.
* **minimum** (float) : The minimum value of the range.
* **maximum** (float) : The maximum value of the range.
* **dtype** (str) : The data type of the normalized array.


**Returns**

* **array**  : The normalized array.


----


### range_tensor
```python
.range_tensor(
   end
)
```

---
Create a tensor with a range of numbers.


**Args**

* **end** (int) : The number at which the range ends.


**Returns**

* **Tensor**  : A tensor containing the numbers in the range.


----


### to_numpy
```python
.to_numpy(
   t
)
```

---
Convert a tensor to a numpy array.


**Args**

* **t** (Tensor) : The tensor to be converted.


**Returns**

* **array**  : The converted numpy array.


----


### to_tensor
```python
.to_tensor(
   x
)
```

---
Convert a value to a tensor.


**Args**

* **x** (various types) : The value to be converted.


**Returns**

* **Tensor**  : The converted tensor.

