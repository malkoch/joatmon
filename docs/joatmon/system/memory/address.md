#


## Address
```python 
Address(
   value, process, default_type = 'uint'
)
```


---
A class used to represent an Address in memory.

Attributes
----------
value : int
The value of the address.
---
    The process that the address belongs to.
    The default type of the address.
    The symbolic name of the address.

Methods
-------
__init__(self, value, process, default_type='uint')
    Initializes a new instance of the Address class.
read(self, of_type=None, max_length=None, errors='raise')
    Reads the value at the address.
write(self, data, of_type=None)
    Writes a value to the address.
symbol(self)
    Gets the symbolic name of the address.
get_instruction(self)
    Gets the instruction at the address.
dump(self, of_type='bytes', size=512, before=32)
    Dumps the memory at the address.


**Methods:**


### .read
```python
.read(
   of_type = None, max_length = None, errors = 'raise'
)
```

---
Reads the value at the address.


**Args**

* **of_type** (str, optional) : The type of the value to read. If not specified, the default type of the address is used.
* **max_length** (int, optional) : The maximum length of the value to read. If not specified, the entire value is read.
* **errors** (str, optional) : The error handling scheme. If 'raise', errors during reading will raise an exception. If 'ignore', errors during reading will be ignored.


**Returns**

* **object**  : The value read from the address.


### .write
```python
.write(
   data, of_type = None
)
```

---
Writes a value to the address.


**Args**

* **data** (object) : The value to write to the address.
* **of_type** (str, optional) : The type of the value to write. If not specified, the default type of the address is used.


**Returns**

* **int**  : The number of bytes written.


### .symbol
```python
.symbol()
```

---
Gets the symbolic name of the address.


**Returns**

* **str**  : The symbolic name of the address.


### .get_instruction
```python
.get_instruction()
```

---
Gets the instruction at the address.


**Returns**

* **str**  : The instruction at the address.


### .dump
```python
.dump(
   of_type = 'bytes', size = 512, before = 32
)
```

---
Dumps the memory at the address.


**Args**

* **of_type** (str, optional) : The type of the memory to dump. Defaults to 'bytes'.
* **size** (int, optional) : The size of the memory to dump. Defaults to 512.
* **before** (int, optional) : The number of bytes before the address to include in the dump. Defaults to 32.


**Returns**

None
