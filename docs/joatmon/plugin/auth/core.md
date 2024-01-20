#


## Auth
```python 
Auth()
```


---
This is the Auth class that inherits from the Plugin class. It is an abstract class that provides
the structure for authentication and authorization methods. The methods in this class should be
implemented in the child classes.


**Methods:**


### .authenticate
```python
.authenticate(
   issuer, audience, expire_at
)
```

---
This is an abstract method that should be implemented in the child classes. It is used to
authenticate a user.


**Args**

* **issuer** (str) : The issuer of the authentication.
* **audience** (str) : The audience of the authentication.
* **expire_at** (datetime) : The expiration date of the authentication.


**Raises**

* **NotImplementedError**  : This method should be implemented in the child classes.


### .authorize
```python
.authorize(
   token, issuer, audience
)
```

---
This is an abstract method that should be implemented in the child classes. It is used to
authorize a user.


**Args**

* **token** (str) : The token used for authorization.
* **issuer** (str) : The issuer of the authorization.
* **audience** (str) : The audience of the authorization.


**Raises**

* **NotImplementedError**  : This method should be implemented in the child classes.

