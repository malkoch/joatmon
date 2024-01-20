#


## LogLevel
```python 
LogLevel()
```


---
LogLevel is an enumeration that defines the different levels of logging.


**Attributes**

* **NotSet** (int) : Level for not setting a logging level.
* **Debug** (int) : Level for debug logging.
* **Info** (int) : Level for information logging.
* **Warning** (int) : Level for warning logging.
* **Error** (int) : Level for error logging.
* **Critical** (int) : Level for critical logging.


----


## LoggerPlugin
```python 
LoggerPlugin(
   level, language, ip
)
```


---
LoggerPlugin is a class that provides logging functionality.


**Attributes**

* **_level** (LogLevel) : The level of logging.
* **language** (str) : The language for logging.
* **ip** (str) : The IP address for logging.



**Methods:**


### ._get_level
```python
._get_level(
   level_str
)
```

---
Get the LogLevel from a string.


**Args**

* **level_str** (str) : The string representation of the LogLevel.


**Returns**

* **LogLevel**  : The LogLevel corresponding to the given string.


### .log
```python
.log(
   log: dict, level: Union[LogLevel, str] = LogLevel.Debug
)
```

---
Log a message at a specified level.


**Args**

* **log** (dict) : The log to be written.
* **level** (Union[LogLevel, str]) : The level at which to log the message.


### .debug
```python
.debug(
   log
)
```

---
Log a debug message.


**Args**

* **log** (dict) : The log to be written.


### .info
```python
.info(
   log
)
```

---
Log an info message.


**Args**

* **log** (dict) : The log to be written.


### .warning
```python
.warning(
   log
)
```

---
Log a warning message.


**Args**

* **log** (dict) : The log to be written.


### .error
```python
.error(
   log
)
```

---
Log an error message.


**Args**

* **log** (dict) : The log to be written.


### .critical
```python
.critical(
   log
)
```

---
Log a critical message.


**Args**

* **log** (dict) : The log to be written.

