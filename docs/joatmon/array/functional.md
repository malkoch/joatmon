#


### prod
```python
.prod(
   inp, axis = None
)
```

---
Calculate the product of all elements in the input list.


**Args**

* **inp** (list) : The input list.
* **axis** (None) : This argument is not used.


**Returns**

* **int**  : The product of all elements in the input list.


----


### unravel_index
```python
.unravel_index(
   index: int, shape: list
)
```

---
Convert a flat index into a multi-dimensional index.


**Args**

* **index** (int) : The flat index.
* **shape** (list) : The shape of the multi-dimensional array.


**Returns**

* **list**  : The multi-dimensional index.


----


### ravel_index
```python
.ravel_index(
   index: list, shape: list
)
```

---
Convert a multi-dimensional index into a flat index.


**Args**

* **index** (list) : The multi-dimensional index.
* **shape** (list) : The shape of the multi-dimensional array.


**Returns**

* **int**  : The flat index.


----


### dim
```python
.dim(
   inp: list
)
```

---
Calculate the dimensionality of the input list.


**Args**

* **inp** (list) : The input list.


**Returns**

* **int**  : The dimensionality of the input list.


----


### flatten
```python
.flatten(
   inp: list
)
```

---
Flatten a multi-dimensional list into a one-dimensional list.


**Args**

* **inp** (list) : The multi-dimensional list.


**Returns**

* **list**  : The flattened list.


----


### size
```python
.size(
   inp: list, axis: int = None
)
```

---
Calculate the size of each dimension of the input list.


**Args**

* **inp** (list) : The input list.
* **axis** (int, optional) : If provided, calculate the size only along this dimension.


**Returns**

* **tuple**  : The size of each dimension of the input list.


----


### reshape
```python
.reshape(
   inp: list, shape: list
)
```

---
Reshape a flat list into a multi-dimensional list.


**Args**

* **inp** (list) : The flat list.
* **shape** (list) : The shape of the multi-dimensional list.


**Returns**

* **list**  : The reshaped list.

