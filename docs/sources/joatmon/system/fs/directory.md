#


## File
```python 
File(
   path
)
```


---
File class that provides the functionality for reading and writing to a file with a read-write lock.


**Attributes**

* **path** (str) : The path of the file.
* **lock** (RWLock) : The read-write lock for the file.

---
Methods:
    write: Writes data to the file.


**Methods:**


### .read
```python
.read()
```

---
Reads the content of the file.

This method uses a read lock to ensure that the file is not modified while it is being read.


**Returns**

* **str**  : The content of the file.


### .write
```python
.write(
   data = ''
)
```

---
Writes data to the file.

This method uses a write lock to ensure that the file is not read while it is being modified.


**Args**

* **data** (str) : The data to be written to the file.

