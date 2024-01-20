#


## Serializable
```python 
Serializable(
   **kwargs
)
```


---
Serializable class for managing serializable objects.

This class provides a way to manage serializable objects, including their conversion to and from various formats such as dictionary, JSON, and bytes.


**Attributes**

* **kwargs**  : Variable length keyword arguments.



**Methods:**


### .keys
```python
.keys()
```

---
Yield the keys of the Serializable object.


**Yields**

* **str**  : The keys of the Serializable object.


### .values
```python
.values()
```

---
Yield the values of the Serializable object.


**Yields**

* **Any**  : The values of the Serializable object.


### .items
```python
.items()
```

---
Yield the key-value pairs of the Serializable object.


**Yields**

* **tuple**  : The key-value pairs of the Serializable object.


### .dict
```python
.dict()
```

---
Convert the Serializable object to a dictionary.


**Returns**

* **dict**  : The dictionary representation of the Serializable object.


### .from_dict
```python
.from_dict(
   cls, dictionary: dict
)
```

---
Create a Serializable object from a dictionary.


**Args**

* **dictionary** (dict) : The dictionary to create the Serializable object from.


**Returns**

* **Serializable**  : The Serializable object created from the dictionary.


### .json
```python
.json()
```

---
Convert the Serializable object to a JSON string.


**Returns**

* **str**  : The JSON string representation of the Serializable object.


### .pretty_json
```python
.pretty_json()
```

---
Convert the Serializable object to a pretty JSON string.


**Returns**

* **str**  : The pretty JSON string representation of the Serializable object.


### .from_json
```python
.from_json(
   cls, string: str
)
```

---
Create a Serializable object from a JSON string.


**Args**

* **string** (str) : The JSON string to create the Serializable object from.


**Returns**

* **Serializable**  : The Serializable object created from the JSON string.


### .bytes
```python
.bytes()
```

---
Convert the Serializable object to bytes.


**Returns**

* **bytes**  : The bytes representation of the Serializable object.


### .from_bytes
```python
.from_bytes(
   cls, value: bytes
)
```

---
Create a Serializable object from bytes.


**Args**

* **value** (bytes) : The bytes to create the Serializable object from.


**Returns**

* **Serializable**  : The Serializable object created from the bytes.


### .snake
```python
.snake()
```

---
Convert the Serializable object to a dictionary with snake case keys.


**Returns**

* **dict**  : The dictionary representation of the Serializable object with snake case keys.


### .pascal
```python
.pascal()
```

---
Convert the Serializable object to a dictionary with Pascal case keys.


**Returns**

* **dict**  : The dictionary representation of the Serializable object with Pascal case keys.

