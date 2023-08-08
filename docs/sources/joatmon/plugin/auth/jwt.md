#


## JWTAuth
```python 
JWTAuth(
   secret: str
)
```




**Methods:**


### .authenticate
```python
.authenticate(
   issuer, audience, expire_at, **kwargs
)
```


### .authorize
```python
.authorize(
   token, issuer, audience
)
```

