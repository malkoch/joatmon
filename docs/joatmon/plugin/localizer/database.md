#


## DatabaseLocalizer
```python 
DatabaseLocalizer(
   database, cls
)
```


---
DatabaseLocalizer class that inherits from the Localizer class. It implements the abstract methods of the Localizer class
using a database for localization operations.


**Attributes**

* **database** (str) : The name of the database to be used for localization.
* **cls** (str) : The class of the documents to be localized.



**Methods:**


### .localize
```python
.localize(
   language, keys
)
```

---
Localize a set of keys to a specified language using a database.

This method reads the keys from the database and localizes them to the specified language. If a key is not found in the database,
it is added with its localized value being the same as the key. If a key is found but does not have a localized value for the
specified language, its localized value is set to the key.


**Args**

* **language** (str) : The language to which the keys should be localized.
* **keys** (list) : The keys to be localized.


**Returns**

* **list**  : The localized keys.

