#


## CustomDictionary
```python 
CustomDictionary(
   data
)
```


---
CustomDictionary class that inherits from the dict class. It provides the functionality for a dictionary with custom methods.


**Attributes**

* **data** (dict) : The data in the dictionary.

---
Methods:
    __setattr__: Sets the value of a specified attribute in the dictionary.


**Methods:**


### .keys
```python
.keys()
```

---
Returns the keys of the dictionary.


**Returns**

* **dict_keys**  : The keys of the dictionary.


### .values
```python
.values()
```

---
Returns the values of the dictionary.


**Returns**

* **dict_values**  : The values of the dictionary.


### .items
```python
.items()
```

---
Returns the items of the dictionary.


**Returns**

* **dict_items**  : The items of the dictionary.


----


### value_to_cls
```python
.value_to_cls(
   value, cls
)
```

---
Converts a value to a specified class.


**Args**

* **value** (any) : The value to be converted.
* **cls** (class) : The class to convert the value to.


**Returns**

* **any**  : The converted value.

