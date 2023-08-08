#


## _RequiredParameter
```python 
_RequiredParameter()
```


---
Singleton class representing a required parameter for an Optimizer.

----


### _calculate_output_dims
```python
._calculate_output_dims(
   input_shape, kernel_shape, padding, stride
)
```


----


### _forward_unimplemented
```python
._forward_unimplemented(
   *inp: Any
)
```

---
Defines the computation performed at every call.

Should be overridden by all subclasses.

.. note::
Although the recipe for forward pass needs to be defined within
this function, one should call the :class:`Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.

----


### typename
```python
.typename(
   o
)
```


----


### indent
```python
.indent(
   string, spaces
)
```


----


### get_enum
```python
.get_enum(
   reduction
)
```


----


### legacy_get_string
```python
.legacy_get_string(
   size_average, reduce, emit_warning = True
)
```


----


### legacy_get_enum
```python
.legacy_get_enum(
   size_average, reduce, emit_warning = True
)
```

