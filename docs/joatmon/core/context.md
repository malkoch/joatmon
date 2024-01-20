#


## CTX
```python 
CTX()
```


---
CTX class for managing context.

This class provides a way to manage context in a global scope.

----


### get_ctx
```python
.get_ctx()
```

---
Get the current context.

This function returns the current context.


**Returns**

* **CTX**  : The current context.


----


### set_ctx
```python
.set_ctx(
   ctx
)
```

---
Set the current context.

This function sets the current context to the provided context.


**Args**

* **ctx** (CTX) : The context to set.


----


### get_value
```python
.get_value(
   name
)
```

---
Get a value from the current context.

This function returns a value from the current context based on the provided name.


**Args**

* **name** (str) : The name of the value to get.


**Returns**

* **Any**  : The value from the current context, or None if the value does not exist.


----


### set_value
```python
.set_value(
   name, value
)
```

---
Set a value in the current context.

This function sets a value in the current context based on the provided name and value.


**Args**

* **name** (str) : The name of the value to set.
* **value** (Any) : The value to set.

