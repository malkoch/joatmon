#


## DatabaseLogger
```python 
DatabaseLogger(
   level: str, database: str, cls, language, ip
)
```


---
DatabaseLogger class that inherits from the LoggerPlugin class. It implements the abstract methods of the LoggerPlugin class
using a database for logging operations.


**Attributes**

* **level** (str) : The level of logging.
* **database** (str) : The name of the database to be used for logging.
* **cls** (str) : The class of the documents to be logged.
* **language** (str) : The language for logging.
* **ip** (str) : The IP address for logging.

