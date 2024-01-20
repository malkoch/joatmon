#


### authorized
```python
.authorized(
   auth, token, issuer
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

