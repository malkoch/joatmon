#


### transaction
```python
.transaction(
   names
)
```

---
Decorator for managing database transactions.

This decorator retrieves one or more database connections from the context and uses them to manage a database transaction. The transaction is started before the function is called and is committed or rolled back after the function is called, depending on whether the function raises an exception.


**Args**

* **names** (list of str) : The names of the database connections in the context.


**Returns**

* **function**  : The decorated function.

