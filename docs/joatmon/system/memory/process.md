#


## ProcessException
```python 
ProcessException()
```


---
A class used to represent a ProcessException.

This exception is raised when there is an error related to the process.

----


## BaseProcess
```python 
BaseProcess(
   *args, **kwargs
)
```


---
A base class used to represent a Process.

Attributes
----------
h_process : object
The handle to the process.
---
    The process ID.
    Whether the process is open.
    The buffer used for reading and writing to the process.
    The length of the buffer.

Methods
-------
__init__(self)
    Initializes a new instance of the BaseProcess class.
__del__(self)
    Deletes the instance of the BaseProcess class.
close(self)
    Closes the process.
iter_region(self, *args, **kwargs)
    Iterates over the regions of the process.
write_bytes(self, address, data)
    Writes bytes to the process.
read_bytes(self, address, size=4)
    Reads bytes from the process.
get_symbolic_name(a)
    Gets the symbolic name of the process.
read(self, address, of_type='uint', max_length=50, errors='raise')
    Reads data from the process.
write(self, address, data, of_type='uint')
    Writes data to the process.


**Methods:**


### .close
```python
.close()
```

---
Closes the process.

### .iter_region
```python
.iter_region(
   *args, **kwargs
)
```

---
Iterates over the regions of the process.

### .write_bytes
```python
.write_bytes(
   address, data
)
```

---
Writes bytes to the process.


**Args**

* **address** (int) : The address to write to.
* **data** (bytes) : The data to write.


**Returns**

* **bool**  : Whether the write was successful.


### .read_bytes
```python
.read_bytes(
   address, size = 4
)
```

---
Reads bytes from the process.


**Args**

* **address** (int) : The address to read from.
* **size** (int, optional) : The number of bytes to read. Defaults to 4.


**Returns**

* **bytes**  : The bytes read from the process.


### .get_symbolic_name
```python
.get_symbolic_name(
   a
)
```

---
Gets the symbolic name of the process.


**Args**

* **a** (int) : The address to get the symbolic name for.


**Returns**

* **str**  : The symbolic name of the process.


### .read
```python
.read(
   address, of_type = 'uint', max_length = 50, errors = 'raise'
)
```

---
Reads data from the process.


**Args**

* **address** (int) : The address to read from.
* **of_type** (str, optional) : The type of data to read. Defaults to 'uint'.
* **max_length** (int, optional) : The maximum length of data to read. Defaults to 50.
* **errors** (str, optional) : The error handling scheme. Defaults to 'raise'.


**Returns**

* **object**  : The data read from the process.


### .write
```python
.write(
   address, data, of_type = 'uint'
)
```

---
Writes data to the process.


**Args**

* **address** (int) : The address to write to.
* **data** (object) : The data to write.
* **of_type** (str, optional) : The type of data to write. Defaults to 'uint'.


**Returns**

* **bool**  : Whether the write was successful.


----


## Process
```python 
Process(
   pid = None, name = None, debug = True
)
```


---
A class used to represent a Process.

Attributes
----------
h_process : object
The handle to the process.
---
    The process ID.
    Whether the process is open.
    The buffer used for reading and writing to the process.
    The length of the buffer.
    The maximum address space for the process.
    The minimum address space for the process.

Methods
-------
__init__(self, pid=None, name=None, debug=True)
    Initializes a new instance of the Process class.
__del__(self)
    Deletes the instance of the Process class.
is_64(self)
    Checks if the process is 64-bit.
list()
    Lists all the processes.
processes_from_name(process_name)
    Gets the processes from the process name.
name_from_process(dw_process_id)
    Gets the process name from the process ID.
_open(self, dw_process_id, debug=False)
    Opens the process with the given process ID.
close(self)
    Closes the process.
_open_from_name(self, process_name, debug=False)
    Opens the process with the given process name.
get_system_info()
    Gets the system information.
get_native_system_info()
    Gets the native system information.
virtual_query_ex(self, lp_address)
    Queries the virtual memory for the given address.
virtual_protect_ex(self, base_address, size, protection)
    Protects the virtual memory for the given base address, size, and protection.
iter_region(self, start_offset=None, end_offset=None, protect=None, optimizations=None)
    Iterates over the regions of the process.
write_bytes(self, address, data)
    Writes bytes to the process.
read_bytes(self, address, size=4, use_nt_wow64_read_virtual_memory64=False)
    Reads bytes from the process.
list_modules(self)
    Lists all the modules of the process.
get_symbolic_name(self, address)
    Gets the symbolic name of the process.
has_module(self, module)
    Checks if the process has the given module.


**Methods:**


### .is_64
```python
.is_64()
```

---
Checks if the process is 64-bit.


**Returns**

* **bool**  : True if the process is 64-bit, False otherwise.


### .list
```python
.list()
```

---
Lists all the processes.


**Returns**

* **list**  : A list of all the processes.


### .processes_from_name
```python
.processes_from_name(
   process_name
)
```

---
Gets the processes from the process name.


**Args**

* **process_name** (str) : The process name.


**Returns**

* **list**  : A list of processes with the given name.


### .name_from_process
```python
.name_from_process(
   dw_process_id
)
```

---
Gets the process name from the process ID.


**Args**

* **dw_process_id** (int) : The process ID.


**Returns**

* **str**  : The process name.


### .close
```python
.close()
```

---
Closes the process.


**Returns**

* **bool**  : True if the process is closed successfully, False otherwise.


### .get_system_info
```python
.get_system_info()
```

---
Gets the system information.


**Returns**

* **object**  : The system information.


### .get_native_system_info
```python
.get_native_system_info()
```

---
Gets the native system information.


**Returns**

* **object**  : The native system information.


### .virtual_query_ex
```python
.virtual_query_ex(
   lp_address
)
```

---
Queries the virtual memory for the given address.


**Args**

* **lp_address** (int) : The address to query.


**Returns**

* **object**  : The memory basic information.


**Raises**

* **ProcessException**  : If there is an error in querying the virtual memory.


### .virtual_protect_ex
```python
.virtual_protect_ex(
   base_address, size, protection
)
```

---
Protects the virtual memory for the given base address, size, and protection.


**Args**

* **base_address** (int) : The base address.
* **size** (int) : The size.
* **protection** (int) : The protection.


**Returns**

* **int**  : The old protection.


**Raises**

* **ProcessException**  : If there is an error in protecting the virtual memory.


### .iter_region
```python
.iter_region(
   start_offset = None, end_offset = None, protect = None, optimizations = None
)
```

---
Iterates over the regions of the process.


**Args**

* **start_offset** (int, optional) : The start offset. Defaults to None.
* **end_offset** (int, optional) : The end offset. Defaults to None.
* **protect** (int, optional) : The protection. Defaults to None.
* **optimizations** (object, optional) : The optimizations. Defaults to None.


**Yields**

* **tuple**  : The offset start and chunk.


### .write_bytes
```python
.write_bytes(
   address, data
)
```

---
Writes bytes to the process.


**Args**

* **address** (int) : The address to write to.
* **data** (bytes) : The data to write.


**Returns**

* **bool**  : True if the write was successful, False otherwise.


**Raises**

* **ProcessException**  : If the process is not open.


### .read_bytes
```python
.read_bytes(
   address, size = 4, use_nt_wow64_read_virtual_memory64 = False
)
```

---
Reads bytes from the process.


**Args**

* **address** (int) : The address to read from.
* **size** (int, optional) : The number of bytes to read. Defaults to 4.
* **use_nt_wow64_read_virtual_memory64** (bool, optional) : Whether to use NtWow64ReadVirtualMemory64. Defaults to False.


**Returns**

* **bytes**  : The bytes read from the process.


**Raises**

* **WindowsError**  : If NtWow64ReadVirtualMemory64 is not available from a 64bit process.
* **WinError**  : If there is an error in reading the bytes.


### .list_modules
```python
.list_modules()
```

---
Lists all the modules of the process.

This method creates a snapshot of the specified processes, as well as the heaps, modules, and threads used by these
processes. It then examines all modules of the process and yields them one by one.


**Yields**

* **ModuleEntry32**  : A module entry from the snapshot of modules for the process.


### .get_symbolic_name
```python
.get_symbolic_name(
   address
)
```

---
Gets the symbolic name of the process.

This method iterates over all modules of the process and checks if the given address falls within the range of
any module. If it does, it returns the module name along with the offset of the address from the base address
of the module.


**Args**

* **address** (int) : The address to get the symbolic name for.


**Returns**

* **str**  : The symbolic name of the process.


### .has_module
```python
.has_module(
   module
)
```

---
Checks if the process has the given module.

This method iterates over all modules of the process and checks if the given module name matches with any of
the module names. If it does, it returns True, otherwise False.


**Args**

* **module** (str) : The name of the module to check.


**Returns**

* **bool**  : True if the process has the module, False otherwise.

