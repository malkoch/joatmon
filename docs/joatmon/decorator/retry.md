#


### retry
```python
.retry(
   times = 5
)
```

---
Decorator for retrying a function call.

This decorator wraps the function in a try-except block. If the function raises an exception, the decorator catches it, prints its message, and retries the function call. The function call is retried a specified number of times.


**Args**

* **times** (int, optional) : The number of times to retry the function call. Defaults to 5.


**Returns**

* **function**  : The decorated function.

