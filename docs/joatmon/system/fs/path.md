#


## Path
```python 
Path(
   path = ''
)
```


---
Path class that provides the functionality for handling file paths.


**Attributes**

* **path** (str) : The path of the file.

---
Methods:
    list: Lists all files and directories in the path.


**Methods:**


### .parent
```python
.parent()
```

---
Returns the parent directory of the path.


**Returns**

* **Path**  : The parent directory of the path.


### .exists
```python
.exists()
```

---
Checks if the path exists.


**Returns**

* **bool**  : True if the path exists, False otherwise.


### .isdir
```python
.isdir()
```

---
Checks if the path is a directory.


**Returns**

* **bool**  : True if the path is a directory, False otherwise.


### .isfile
```python
.isfile()
```

---
Checks if the path is a file.


**Returns**

* **bool**  : True if the path is a file, False otherwise.


### .mkdir
```python
.mkdir()
```

---
Creates a directory at the path.


**Returns**

None

### .touch
```python
.touch()
```

---
Creates a file at the path.


**Returns**

None

### .list
```python
.list()
```

---
Lists all files and directories in the path.


**Returns**

* **list**  : A list of all files and directories in the path.

