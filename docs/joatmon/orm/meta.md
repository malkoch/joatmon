#


## Meta
```python 
Meta()
```


---
Metaclass for ORM system. It provides methods to access fields, constraints, and indexes of a class.


**Attributes**

* **__collection__** (str) : The collection name in the database.
* **structured** (bool) : Whether the class is structured.
* **force** (bool) : Whether to force the structure.
* **qb**  : The query builder for the class.



**Methods:**


### .fields
```python
.fields(
   cls, predicate = lambdax: True
)
```

---
Gets the fields of the class.


**Args**

* **predicate** (callable) : A function that determines which fields to include.


**Returns**

* **dict**  : A dictionary of the fields of the class.


### .constraints
```python
.constraints(
   cls, predicate = lambdax: True
)
```

---
Gets the constraints of the class.


**Args**

* **predicate** (callable) : A function that determines which constraints to include.


**Returns**

* **dict**  : A dictionary of the constraints of the class.


### .indexes
```python
.indexes(
   cls, predicate = lambdax: True
)
```

---
Gets the indexes of the class.


**Args**

* **predicate** (callable) : A function that determines which indexes to include.


**Returns**

* **dict**  : A dictionary of the indexes of the class.


### .query
```python
.query(
   cls
)
```

---
Gets the query builder for the class.


**Returns**

The query builder for the class.

----


### normalize_kwargs
```python
.normalize_kwargs(
   meta, **kwargs
)
```

---
Normalizes the keyword arguments to match the fields of the class.


**Args**

* **meta** (Meta) : The metaclass of the class.
* **kwargs**  : The keyword arguments to normalize.


**Returns**

* **dict**  : A dictionary of the normalized keyword arguments.

