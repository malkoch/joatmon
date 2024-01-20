#


### cached
```python
.cached(
   cache, duration
)
```

---
Decorator for authorizing a function call.

This decorator retrieves the current token and issuer from the context, and uses them to authorize the function call. If the authorization is successful, the function is called; otherwise, an exception is raised.


**Args**

* **auth** (str) : The name of the authorizer in the context.
* **token** (str) : The name of the token in the context.
* **issuer** (str) : The name of the issuer in the context.


**Returns**

* **function**  : The decorated function.


----


### remove
```python
.remove(
   cache, regex
)
```

---
Decorator for removing entries from a cache.

This decorator retrieves a cache from the context and uses it to remove entries that match a specified regular expression. After the entries are removed, the function is called.


**Args**

* **cache** (str) : The name of the cache in the context.
* **regex** (str) : The regular expression to match entries against.


**Returns**

* **function**  : The decorated function.

