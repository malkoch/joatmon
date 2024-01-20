#


### deprecated
```python
.deprecated(
   reason: str
)
```

---
Decorator for marking functions as deprecated.

This decorator emits a warning when the decorated function is called. The warning includes the name of the function and the reason for its deprecation.


**Args**

* **reason** (str) : The reason for the function's deprecation.


**Returns**

* **Callable**  : The decorated function.

