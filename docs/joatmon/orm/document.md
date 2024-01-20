#


## Document
```python 
Document(
   **kwargs
)
```


---
Base class for all documents in the ORM system.


**Attributes**

* **__metaclass__** (Meta) : The metaclass that contains the document's metadata.



**Methods:**


### .keys
```python
.keys()
```

---
Gets the names of the fields in the document.


**Returns**

* **list**  : The names of the fields in the document.


### .values
```python
.values()
```

---
Gets the values of the fields in the document.


**Returns**

* **list**  : The values of the fields in the document.


### .validate
```python
.validate()
```

---
Validates the document.


**Returns**

* **dict**  : A dictionary containing the validated fields and their values.


**Raises**

* **ValueError**  : If the document is not valid.


----


### create_new_type
```python
.create_new_type(
   meta, subclasses
)
```

---
Creates a new type with the specified metaclass and subclasses.


**Args**

* **meta** (Meta) : The metaclass for the new type.
* **subclasses** (tuple) : The subclasses for the new type.


**Returns**

* **type**  : The new type.

