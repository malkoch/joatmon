#


## ValidationException
```python 
ValidationException()
```



----


## Constraint
```python 
Constraint(
   field, validator = None
)
```




**Methods:**


### .create
```python
.create(
   constraint_type, **kwargs
)
```


### .check
```python
.check(
   obj
)
```


----


## LengthConstraint
```python 
LengthConstraint(
   field, min_length = None, max_length = None
)
```



----


## IntegerValueConstraint
```python 
IntegerValueConstraint(
   field, min_value = None, max_value = None
)
```



----


## PrimaryKeyConstraint
```python 
PrimaryKeyConstraint(
   field
)
```



----


## ForeignKeyConstrain
```python 
ForeignKeyConstrain()
```



----


## UniqueConstraint
```python 
UniqueConstraint(
   field
)
```



----


## CustomConstraint
```python 
CustomConstraint(
   field, validator = lambdax: True
)
```


