#


## Worker
```python 
Worker(
   pid = None, name = None, debug = True
)
```


---
A class used to represent a Worker.

Attributes
----------
process : object
The process that the worker belongs to.

---
Methods
-------
__init__(self, pid=None, name=None, debug=True)
    Initializes a new instance of the Worker class.
__enter__(self)
    Enters the context of the Worker class.
__exit__(self)
    Exits the context of the Worker class.
address(self, value, default_type='uint')
    Returns an Address object with the given value and default type.
memory_replace_unicode(self, regex, replace)
    Replaces all occurrences of the regex with the replace in the memory.
memory_replace(self, regex, replace)
    Replaces all occurrences of the regex with the replace in the memory.
memory_search_unicode(self, regex)
    Searches for the regex in the memory.
group_search(self, group, start_offset=None, end_offset=None)
    Searches for the group in the memory.
search_address(self, search)
    Searches for the address in the memory.
parse_re_function(self, byte_array, value, offset)
    Parses the byte array using the regex function.
parse_float_function(self, byte_array, value, offset)
    Parses the byte array using the float function.
parse_named_groups_function(byte_array, value, _)
    Parses the byte array using the named groups function.
parse_groups_function(byte_array, value, _)
    Parses the byte array using the groups function.
parse_any_function(self, byte_array, value, offset)
    Parses the byte array using any function.
memory_search(
        self,
        value,
        of_type='match',
        protect=PAGE_READWRITE | PAGE_READONLY,
        optimizations=None,
        start_offset=None,
        end_offset=None,
)
    Searches for the value in the memory.


**Methods:**


### .address
```python
.address(
   value, default_type = 'uint'
)
```

---
Returns an Address object with the given value and default type.


**Args**

* **value** (int) : The value of the address.
* **default_type** (str, optional) : The default type of the address. Defaults to 'uint'.


**Returns**

* **Address**  : An Address object with the given value and default type.


### .memory_replace_unicode
```python
.memory_replace_unicode(
   regex, replace
)
```

---
Replaces all occurrences of the regex with the replace in the memory.


**Args**

* **regex** (str) : The regex to replace.
* **replace** (str) : The string to replace the regex with.


**Returns**

* **bool**  : Whether the replacement was successful.


### .memory_replace
```python
.memory_replace(
   regex, replace
)
```

---
Replaces all occurrences of the regex with the replace in the memory.


**Args**

* **regex** (str) : The regex to replace.
* **replace** (str) : The string to replace the regex with.


**Returns**

* **bool**  : Whether the replacement was successful.


### .memory_search_unicode
```python
.memory_search_unicode(
   regex
)
```

---
Searches for the regex in the memory.


**Args**

* **regex** (str) : The regex to search for.


**Yields**

* **tuple**  : A tuple containing the name and address of the regex.


### .group_search
```python
.group_search(
   group, start_offset = None, end_offset = None
)
```

---
Searches for the group in the memory.


**Args**

* **group** (list) : The group to search for.
* **start_offset** (int, optional) : The start offset of the search. Defaults to None.
* **end_offset** (int, optional) : The end offset of the search. Defaults to None.


**Returns**

* **list**  : The addresses of the group in the memory.


### .search_address
```python
.search_address(
   search
)
```

---
Searches for the address in the memory.


**Args**

* **search** (int) : The address to search for.


**Yields**

* **int**  : The address in the memory.


### .parse_re_function
```python
.parse_re_function(
   byte_array, value, offset
)
```

---
Parses the byte array using the regex function.


**Args**

* **byte_array** (bytes) : The byte array to parse.
* **value** (str) : The value to search for in the byte array.
* **offset** (int) : The offset of the byte array.


**Yields**

* **tuple**  : A tuple containing the name and address of the value in the byte array.


### .parse_float_function
```python
.parse_float_function(
   byte_array, value, offset
)
```

---
Parses the byte array using the float function.


**Args**

* **byte_array** (bytes) : The byte array to parse.
* **value** (float) : The value to search for in the byte array.
* **offset** (int) : The offset of the byte array.


**Yields**

* **Address**  : An Address object with the offset and type 'float'.


### .parse_named_groups_function
```python
.parse_named_groups_function(
   byte_array, value, _
)
```

---
Parses the byte array using the named groups function.


**Args**

* **byte_array** (bytes) : The byte array to parse.
* **value** (str) : The value to search for in the byte array.
* **_** (None) : Unused parameter.


**Yields**

* **tuple**  : A tuple containing the name and group dictionary of the value in the byte array.


### .parse_groups_function
```python
.parse_groups_function(
   byte_array, value, _
)
```

---
Parses the byte array using the groups function.


**Args**

* **byte_array** (bytes) : The byte array to parse.
* **value** (str) : The value to search for in the byte array.
* **_** (None) : Unused parameter.


**Yields**

* **tuple**  : A tuple containing the name and groups of the value in the byte array.


### .parse_any_function
```python
.parse_any_function(
   byte_array, value, offset
)
```

---
Parses the byte array using any function.


**Args**

* **byte_array** (bytes) : The byte array to parse.
* **value** (str) : The value to search for in the byte array.
* **offset** (int) : The offset of the byte array.


**Yields**

* **Address**  : An Address object with the offset and type 'bytes'.


### .memory_search
```python
.memory_search(
   value, of_type = 'match', protect = PAGE_READWRITE|PAGE_READONLY,
   optimizations = None, start_offset = None, end_offset = None
)
```

---
Searches for the value in the memory.


**Args**

* **value** (str) : The value to search for.
* **of_type** (str, optional) : The type of the value. Defaults to 'match'.
* **protect** (int, optional) : The protection of the memory. Defaults to PAGE_READWRITE | PAGE_READONLY.
* **optimizations** (None, optional) : The optimizations to use. Defaults to None.
* **start_offset** (int, optional) : The start offset of the search. Defaults to None.
* **end_offset** (int, optional) : The end offset of the search. Defaults to None.


**Yields**

* **tuple**  : A tuple containing the name and address of the value in the memory.

