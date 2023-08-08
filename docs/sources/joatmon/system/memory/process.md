#


## ProcessException
```python 
ProcessException()
```



----


## BaseProcess
```python 
BaseProcess(
   *args, **kwargs
)
```




**Methods:**


### .close
```python
.close()
```


### .iter_region
```python
.iter_region(
   *args, **kwargs
)
```


### .write_bytes
```python
.write_bytes(
   address, data
)
```


### .read_bytes
```python
.read_bytes(
   address, size = 4
)
```


### .get_symbolic_name
```python
.get_symbolic_name(
   a
)
```


### .read
```python
.read(
   address, of_type = 'uint', max_length = 50, errors = 'raise'
)
```


### .write
```python
.write(
   address, data, of_type = 'uint'
)
```


----


## Process
```python 
Process(
   pid = None, name = None, debug = True
)
```




**Methods:**


### .is_64
```python
.is_64()
```


### .list
```python
.list()
```


### .processes_from_name
```python
.processes_from_name(
   process_name
)
```


### .name_from_process
```python
.name_from_process(
   dw_process_id
)
```


### .close
```python
.close()
```


### .get_system_info
```python
.get_system_info()
```


### .get_native_system_info
```python
.get_native_system_info()
```


### .virtual_query_ex
```python
.virtual_query_ex(
   lp_address
)
```


### .virtual_protect_ex
```python
.virtual_protect_ex(
   base_address, size, protection
)
```


### .iter_region
```python
.iter_region(
   start_offset = None, end_offset = None, protect = None, optimizations = None
)
```


### .write_bytes
```python
.write_bytes(
   address, data
)
```


### .read_bytes
```python
.read_bytes(
   address, size = 4, use_nt_wow64_read_virtual_memory64 = False
)
```


### .list_modules
```python
.list_modules()
```


### .get_symbolic_name
```python
.get_symbolic_name(
   address
)
```


### .has_module
```python
.has_module(
   module
)
```

