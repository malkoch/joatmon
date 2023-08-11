#


## Worker
```python 
Worker(
   pid = None, name = None, debug = True
)
```




**Methods:**


### .address
```python
.address(
   value, default_type = 'uint'
)
```


### .memory_replace_unicode
```python
.memory_replace_unicode(
   regex, replace
)
```


### .memory_replace
```python
.memory_replace(
   regex, replace
)
```


### .memory_search_unicode
```python
.memory_search_unicode(
   regex
)
```


### .group_search
```python
.group_search(
   group, start_offset = None, end_offset = None
)
```


### .search_address
```python
.search_address(
   search
)
```


### .parse_re_function
```python
.parse_re_function(
   byte_array, value, offset
)
```


### .parse_float_function
```python
.parse_float_function(
   byte_array, value, offset
)
```


### .parse_named_groups_function
```python
.parse_named_groups_function(
   byte_array, value, _
)
```


### .parse_groups_function
```python
.parse_groups_function(
   byte_array, value, _
)
```


### .parse_any_function
```python
.parse_any_function(
   byte_array, value, offset
)
```


### .memory_search
```python
.memory_search(
   value, of_type = 'match', protect = PAGE_READWRITE|PAGE_READONLY,
   optimizations = None, start_offset = None, end_offset = None
)
```

