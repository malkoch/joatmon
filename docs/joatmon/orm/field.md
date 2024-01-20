#


## Field
```python 
Field(
   dtype: typing.Union[type, typing.List, typing.Tuple], nullable: bool = True,
   default = None, primary: bool = False, hash_: bool = False, resource: bool = False,
   fields: dict = None
)
```


---
Base class for all fields in the ORM system.


**Attributes**

* **dtype** (type) : The data type of the field.
* **nullable** (bool) : Whether the field can be null.
* **primary** (bool) : Whether the field is a primary key.
* **hash_** (bool) : Whether the field is a hash field.
* **fields** (dict) : The sub-fields of the field, if it is a complex field.


----


### get_converter
```python
.get_converter(
   field: Field
)
```

---
Returns a converter function for the given field.


**Args**

* **field** (Field) : The field for which to get a converter.


**Returns**

* **callable**  : A function that can convert values to the field's data type.

