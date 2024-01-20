#


### authorized
```python
.authorized(
   auth, token, issuer
)
```

---
Decorator for authorizing function calls.

This decorator retrieves an authorizer, a token, and an issuer from the context. It uses the authorizer to authorize the function call with the token, the issuer, and the audience. The audience is the fully qualified name of the function.


**Args**

* **auth** (str) : The name of the authorizer in the context.
* **token** (str) : The name of the token in the context.
* **issuer** (str) : The name of the issuer in the context.


**Returns**

* **function**  : The decorated function.

