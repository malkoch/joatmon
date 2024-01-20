#


## ValidationException
```python 
ValidationException()
```


---
Exception raised for errors in the validation process.


**Attributes**

message -- explanation of the error

----


## Constraint
```python 
Constraint(
   field, validator = None
)
```


---
Base class for all constraints.


**Attributes**

* **field** (str) : The field to which the constraint applies.
* **validator** (callable) : A function that validates the field's value.



**Methods:**


### .create
```python
.create(
   constraint_type, **kwargs
)
```

---
Factory method for creating constraints.


**Args**

* **constraint_type** (str) : The type of constraint to create.
* **kwargs**  : Additional keyword arguments for the constraint's constructor.


**Returns**

* **Constraint**  : A new constraint of the specified type.


### .check
```python
.check(
   obj
)
```

---
Checks whether the constraint is satisfied.


**Args**

* **obj**  : The object to check.


**Returns**

* **bool**  : True if the constraint is satisfied, False otherwise.


**Raises**

* **ValidationException**  : If the constraint is not satisfied.


----


## LengthConstraint
```python 
LengthConstraint(
   field, min_length = None, max_length = None
)
```


---
Constraint that checks whether a field's value has a valid length.


**Attributes**

* **min_length** (int) : The minimum valid length. None if there is no minimum.
* **max_length** (int) : The maximum valid length. None if there is no maximum.


----


## IntegerValueConstraint
```python 
IntegerValueConstraint(
   field, min_value = None, max_value = None
)
```


---
Constraint that checks whether a field's value is within a valid range.


**Attributes**

* **min_value** (int) : The minimum valid value. None if there is no minimum.
* **max_value** (int) : The maximum valid value. None if there is no maximum.


----


## PrimaryKeyConstraint
```python 
PrimaryKeyConstraint(
   field
)
```


---
Constraint that checks whether a field's value is a valid primary key.

----


## ForeignKeyConstraint
```python 
ForeignKeyConstraint()
```


---
Constraint that checks whether a field's value is a valid foreign key.

----


## UniqueConstraint
```python 
UniqueConstraint(
   field
)
```


---
Constraint that checks whether a field's value is unique.

----


## CustomConstraint
```python 
CustomConstraint(
   field, validator = lambdax: True
)
```


---
Constraint that checks whether a field's value satisfies a custom condition.


**Attributes**

* **validator** (callable) : A function that validates the field's value.

