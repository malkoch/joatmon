#


## Meta
```python 
Meta()
```


---
Metaclass for Enum class. It extends EnumMeta and overrides the __contains__ method.

----


## Enum
```python 
Enum()
```


---
Enum class that extends IntEnum. It provides additional methods for comparison and string representation.


**Methods:**


### .parse
```python
.parse(
   value: str
)
```

---
Parses a string into an Enum member.


**Args**

* **value** (str) : The string to parse.


**Returns**

* **Enum**  : The parsed Enum member.


**Raises**

* **ValueError**  : If the string cannot be parsed into an Enum member.

