#


## _RequiredParameter
```python 
_RequiredParameter()
```


---
Singleton class representing a required parameter for an Optimizer.

----


### _calculate_input_dims
```python
._calculate_input_dims(
   output_shape, kernel_shape, padding, stride
)
```

---
Calculates the dimensions of the input tensor given the output shape, kernel shape, padding, and stride.


**Args**

* **output_shape** (tuple) : The shape of the output tensor.
* **kernel_shape** (tuple) : The shape of the kernel tensor.
* **padding** (tuple) : The padding applied to the input tensor.
* **stride** (int) : The stride applied to the input tensor.


**Returns**

* **tuple**  : The dimensions of the input tensor.


----


### _calculate_output_dims
```python
._calculate_output_dims(
   input_shape, kernel_shape, padding, stride
)
```

---
Calculates the dimensions of the output tensor given the input shape, kernel shape, padding, and stride.


**Args**

* **input_shape** (tuple) : The shape of the input tensor.
* **kernel_shape** (tuple) : The shape of the kernel tensor.
* **padding** (tuple) : The padding applied to the input tensor.
* **stride** (int) : The stride applied to the input tensor.


**Returns**

* **tuple**  : The dimensions of the output tensor.


----


### _forward_unimplemented
```python
._forward_unimplemented(
   *inp: Any
)
```

---
Raises a NotImplementedError indicating that the forward method needs to be implemented in subclasses.


**Args**

* **inp** (Any) : Variable length argument list.


**Raises**

* **NotImplementedError**  : This method needs to be implemented in subclasses.


----


### typename
```python
.typename(
   o
)
```

---
Returns the type name of the object.


**Args**

* **o** (object) : The object to get the type name of.


**Returns**

* **str**  : The type name of the object.


----


### indent
```python
.indent(
   string, spaces
)
```

---
Indents a string by a given number of spaces.


**Args**

* **string** (str) : The string to indent.
* **spaces** (int) : The number of spaces to indent the string by.


**Returns**

* **str**  : The indented string.


----


### get_enum
```python
.get_enum(
   reduction
)
```

---
Returns the enum value for a given reduction type.


**Args**

* **reduction** (str) : The reduction type.


**Returns**

* **int**  : The enum value for the reduction type.


**Raises**

* **ValueError**  : If the reduction type is not valid.


----


### legacy_get_string
```python
.legacy_get_string(
   size_average, reduce, emit_warning = True
)
```

---
Returns the string value for a given size_average and reduce type.


**Args**

* **size_average** (bool) : The size_average type.
* **reduce** (bool) : The reduce type.
* **emit_warning** (bool, optional) : Whether to emit a warning. Defaults to True.


**Returns**

* **str**  : The string value for the size_average and reduce type.


----


### legacy_get_enum
```python
.legacy_get_enum(
   size_average, reduce, emit_warning = True
)
```

---
Returns the enum value for a given size_average and reduce type.


**Args**

* **size_average** (bool) : The size_average type.
* **reduce** (bool) : The reduce type.
* **emit_warning** (bool, optional) : Whether to emit a warning. Defaults to True.


**Returns**

* **int**  : The enum value for the size_average and reduce type.

