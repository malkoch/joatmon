#


## ModuleEntry32
```python 
ModuleEntry32()
```


---
A class used to represent a module entry in a 32-bit system.

Attributes
----------
dwSize : ctypes.c_ulong
The size of the structure.
---
    The identifier of the module.
    The identifier of the process.
    The global count of the usage.
    The process count of the usage.
    The base address of the module in the context of the owning process.
    The size of the module, in bytes.
    A handle to the module in the context of the owning process.
    The module name.
    The module path.

----


## MemoryBasicInformation
```python 
MemoryBasicInformation()
```


---
A class used to represent basic memory information.

Attributes
----------
BaseAddress : ctypes.c_void_p
A pointer to the base address of the region of pages.
---
    A pointer to the base address of a range of pages allocated by the VirtualAlloc function.
    The memory protection option when the region was initially allocated.
    The size of the region beginning at the base address in which all pages have identical attributes, in bytes.
    The state of the pages in the region (free, reserve, or commit).
    The access protection of the pages in the region.
    The type of pages in the region (private, mapped, or image).

----


## SystemInfo
```python 
SystemInfo()
```


---
A class used to represent system information.

Attributes
----------
wProcessorArchitecture : wintypes.WORD
The architecture of the processor.
---
    Reserved.
    The page size, in bytes.
    A pointer to the lowest memory address accessible to applications and dynamic-link libraries (DLLs).
    A pointer to the highest memory address accessible to applications and DLLs.
    A mask representing the set of processors configured into the system.
    The number of logical processors in the current group.
    The type of processor.
    The granularity for the starting address at which virtual memory can be allocated.
    The architecture-dependent processor level.
    The architecture-dependent revision of the processor.

----


## SecurityDescriptor
```python 
SecurityDescriptor()
```


---
A class used to represent a security descriptor.

Attributes
----------
SID : wintypes.DWORD
The security identifier (SID) for the security descriptor.
---
    The primary group SID for the security descriptor.
    The discretionary access control list (DACL) for the security descriptor.
    The system access control list (SACL) for the security descriptor.
    Test field for the security descriptor.

----


## Th32csClass
```python 
Th32csClass()
```


---
A class used to represent a snapshot of the specified processes, as well as the heaps, modules, and threads used by these processes.

Attributes
----------
INHERIT : int
Indicates that the returned handle can be inherited by child processes of the current process.
---
    Includes all heaps of the specified process in the snapshot.
    Includes all modules of the specified process in the snapshot.
    Includes all 32-bit modules of the specified process in the snapshot.
    Includes all processes in the system in the snapshot.
    Includes all threads in the system in the snapshot.
    Includes all of the above in the snapshot.

----


### type_unpack
```python
.type_unpack(
   of_type
)
```

---
Unpacks the given type into a structure type and structure length.


**Args**

* **of_type** (str) : The type to unpack.


**Returns**

* **tuple**  : A tuple containing the structure type and structure length.


**Raises**

* **TypeError**  : If the given type is unknown.


----


### hex_dump
```python
.hex_dump(
   data, address = 0, prefix = '', of_type = 'bytes'
)
```

---
Dumps the given data in hexadecimal format.


**Args**

* **data** (bytes) : The data to dump.
* **address** (int, optional) : The starting address of the data. Defaults to 0.
* **prefix** (str, optional) : The prefix to add to each line of the dump. Defaults to ''.
* **of_type** (str, optional) : The type of the data. Defaults to 'bytes'.


**Returns**

* **str**  : The hexadecimal dump of the data.

