#


## Relation
```python 
Relation(
   relation = ''
)
```


---
Class representing a relation in the ORM system.

A relation represents a connection between two collections in the database.
It is defined by a string in the format "collection1.field1(n1)->collection2.field2(n2)",
where "n1" and "n2" are the cardinalities of the relation.


**Attributes**

* **local_collection** (str) : The name of the local collection.
* **local_field** (str) : The name of the local field.
* **local_relation** (str) : The cardinality of the local relation.
* **foreign_collection** (str) : The name of the foreign collection.
* **foreign_field** (str) : The name of the foreign field.
* **foreign_relation** (str) : The cardinality of the foreign relation.

