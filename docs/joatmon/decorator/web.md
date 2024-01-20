#


### get
```python
.get(
   func
)
```

---
Decorator for HTTP GET method.

This decorator marks the function as a handler for HTTP GET requests.


**Args**

* **func** (function) : The function to be decorated.


**Returns**

* **function**  : The decorated function.


----


### post
```python
.post(
   func
)
```

---
Decorator for HTTP POST method.

This decorator marks the function as a handler for HTTP POST requests.


**Args**

* **func** (function) : The function to be decorated.


**Returns**

* **function**  : The decorated function.


----


### incoming
```python
.incoming(
   case, json, arg, form
)
```

---
Decorator for handling incoming requests.

This decorator retrieves the request data from the context and updates the function's keyword arguments with it.


**Args**

* **case** (str) : The name of the case in the context.
* **json** (str) : The name of the JSON data in the context.
* **arg** (str) : The name of the arguments in the context.
* **form** (str) : The name of the form data in the context.


**Returns**

* **function**  : The decorated function.


----


### wrap
```python
.wrap(
   func
)
```

---
Decorator for wrapping function calls.

This decorator wraps the function call in a try-except block. If the function call is successful, it returns a dictionary with the result and a success status. If the function call raises a CoreException, it returns a dictionary with an error message and a failure status.


**Args**

* **func** (function) : The function to be decorated.


**Returns**

* **function**  : The decorated function.


----


### outgoing
```python
.outgoing(
   case
)
```

---
Decorator for handling outgoing responses.

This decorator retrieves the case from the context and converts the function's return value to that case.


**Args**

* **case** (str) : The name of the case in the context.


**Returns**

* **function**  : The decorated function.


----


### ip_limit
```python
.ip_limit(
   interval, cache, ip
)
```

---
Decorator for limiting requests per IP.

This decorator retrieves the cache and the IP from the context. It limits the number of requests from the IP to one per specified interval.


**Args**

* **interval** (int) : The interval in seconds between requests.
* **cache** (str) : The name of the cache in the context.
* **ip** (str) : The name of the IP in the context.


**Returns**

* **function**  : The decorated function.


----


### limit
```python
.limit(
   interval, cache
)
```

---
Decorator for limiting requests.

This decorator retrieves the cache from the context. It limits the number of requests to one per specified interval.


**Args**

* **interval** (int) : The interval in seconds between requests.
* **cache** (str) : The name of the cache in the context.


**Returns**

* **function**  : The decorated function.

