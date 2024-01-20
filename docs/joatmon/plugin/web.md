#


## ValuePlugin
```python 
ValuePlugin(
   key
)
```


---
ValuePlugin class that inherits from the Plugin class. It provides the functionality for setting and getting values in the context.


**Attributes**

* **key** (str) : The key for the value in the context.

---
Methods:
    get: Gets a value from the context.


**Methods:**


### .set
```python
.set(
   value
)
```

---
Sets a value in the context.

This method uses the joatmon.core.context.set_value method to set a value in the context.


**Args**

* **value** (any) : The value to be set in the context.


### .get
```python
.get()
```

---
Gets a value from the context.

This method uses the joatmon.core.context.get_value method to get a value from the context.


**Returns**

* **any**  : The value from the context.

