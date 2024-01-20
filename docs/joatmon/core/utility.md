#


## JSONEncoder
```python 
JSONEncoder()
```


---
A JSON encoder that can handle more Python data types than the standard json.JSONEncoder.

This encoder can handle datetime, date, time, timedelta, uuid.UUID, bytes, and callable objects in addition to the types that json.JSONEncoder can handle.


**Methods:**


### .default
```python
.default(
   obj
)
```

---
Implement this method in a subclass such that it returns a serializable object for `obj`, or calls the base implementation (to raise a TypeError).


**Args**

* **obj** (Any) : The object to convert to a serializable object.


**Returns**

* **Any**  : A serializable object.


----


## JSONDecoder
```python 
JSONDecoder(
   *args, **kwargs
)
```


---
A JSON decoder that uses a custom object hook.

This decoder uses an object hook that simply returns the input dictionary without any modifications.


**Methods:**


### .object_hook
```python
.object_hook(
   d
)
```

---
Implement this method in a subclass such that it returns a Python object for `d`.


**Args**

* **d** (dict) : The dictionary to convert to a Python object.


**Returns**

* **Any**  : A Python object.


----


### empty_object_id
```python
.empty_object_id()
```

---
Convert the Serializable object to a dictionary with Pascal case keys.


**Returns**

* **dict**  : The dictionary representation of the Serializable object with Pascal case keys.


----


### new_object_id
```python
.new_object_id()
```

---
Generate a new UUID.


**Returns**

* **UUID**  : A new UUID.


----


### current_time
```python
.current_time()
```

---
Get the current datetime.


**Returns**

* **datetime**  : The current datetime.


----


### new_nickname
```python
.new_nickname()
```

---
Generate a new random nickname.


**Returns**

* **str**  : A new random nickname.


----


### new_password
```python
.new_password()
```

---
Generate a new random password.


**Returns**

* **str**  : A new random password.


----


### mail_validator
```python
.mail_validator(
   email
)
```

---
Validate an email address.


**Args**

* **email** (str) : The email address to validate.


**Returns**

* **bool**  : True if the email address is valid, False otherwise.


----


### ip_validator
```python
.ip_validator(
   ip
)
```

---
Validate an IP address.


**Args**

* **ip** (str) : The IP address to validate.


**Returns**

* **bool**  : True if the IP address is valid, False otherwise.


----


### to_snake_string
```python
.to_snake_string(
   string: str
)
```

---
Convert a string to snake case.


**Args**

* **string** (str) : The string to convert.


**Returns**

* **str**  : The string in snake case.


----


### to_pascal_string
```python
.to_pascal_string(
   string: str
)
```

---
Convert a string to Pascal case.


**Args**

* **string** (str) : The string to convert.


**Returns**

* **str**  : The string in Pascal case.


----


### to_upper_string
```python
.to_upper_string(
   string: str
)
```

---
Convert a string to upper case.


**Args**

* **string** (str) : The string to convert.


**Returns**

* **str**  : The string in upper case.


----


### to_lower_string
```python
.to_lower_string(
   string: str
)
```

---
Convert a string to lower case.


**Args**

* **string** (str) : The string to convert.


**Returns**

* **str**  : The string in lower case.


----


### to_title
```python
.to_title(
   string: str
)
```

---
Convert a string to title case.


**Args**

* **string** (str) : The string to convert.


**Returns**

* **str**  : The string in title case.


----


### to_enumerable
```python
.to_enumerable(
   value, string = False
)
```

---
Convert a value to an enumerable.


**Args**

* **value** (Any) : The value to convert.
* **string** (bool) : If True, convert the value to a string.


**Returns**

* **Any**  : The value as an enumerable.


----


### to_case
```python
.to_case(
   case, value, key = None, convert_value = False
)
```

---
Convert a value to a specific case.


**Args**

* **case** (str) : The case to convert to.
* **value** (Any) : The value to convert.
* **key** (str, optional) : The key to convert. Defaults to None.
* **convert_value** (bool, optional) : If True, convert the value to a string. Defaults to False.


**Returns**

* **Any**  : The value converted to the specified case.


----


### get_function_args
```python
.get_function_args(
   func, *args
)
```

---
Get the arguments of a function.


**Args**

* **func** (function) : The function to get the arguments of.
* **args**  : The arguments of the function.


**Returns**

* **tuple**  : The arguments of the function.


----


### get_function_kwargs
```python
.get_function_kwargs(
   func, **kwargs
)
```

---
Get the keyword arguments of a function.


**Args**

* **func** (function) : The function to get the keyword arguments of.
* **kwargs**  : The keyword arguments of the function.


**Returns**

* **dict**  : The keyword arguments of the function.


----


### to_hash
```python
.to_hash(
   func, *args, **kwargs
)
```

---
Generate a hash for a function and its arguments.


**Args**

* **func** (function) : The function to generate a hash for.
* **args**  : The arguments of the function.
* **kwargs**  : The keyword arguments of the function.


**Returns**

* **str**  : The hash of the function and its arguments.


----


### get_converter
```python
.get_converter(
   kind: type
)
```

---
Get a converter for a specific type.


**Args**

* **kind** (type) : The type to get a converter for.


**Returns**

* **function**  : The converter for the specified type.


----


### to_list
```python
.to_list(
   items
)
```

---
Convert an iterable to a list.


**Args**

* **items** (iterable) : The iterable to convert.


**Returns**

* **list**  : The list representation of the iterable.


----


### to_list_async
```python
.to_list_async(
   items
)
```

---
Asynchronously convert an iterable to a list.


**Args**

* **items** (iterable) : The iterable to convert.


**Returns**

* **list**  : The list representation of the iterable.


----


### first
```python
.first(
   items
)
```

---
Get the first item of an iterable.


**Args**

* **items** (iterable) : The iterable to get the first item from.


**Returns**

* **Any**  : The first item of the iterable, or None if the iterable is empty.


----


### first_async
```python
.first_async(
   items
)
```

---
Asynchronously get the first item of an iterable.


**Args**

* **items** (iterable) : The iterable to get the first item from.


**Returns**

* **Any**  : The first item of the iterable, or None if the iterable is empty.


----


### single
```python
.single(
   items
)
```

---
Get the single item of an iterable.


**Args**

* **items** (iterable) : The iterable to get the single item from.


**Returns**

* **Any**  : The single item of the iterable, or None if the iterable is empty or contains more than one item.


----


### single_async
```python
.single_async(
   items
)
```

---
Asynchronously get the single item of an iterable.


**Args**

* **items** (iterable) : The iterable to get the single item from.


**Returns**

* **Any**  : The single item of the iterable, or None if the iterable is empty or contains more than one item.


----


### pretty_printer
```python
.pretty_printer(
   headers, m = None
)
```

---
Create a pretty printer for a list of headers.


**Args**

* **headers** (list) : The headers to pretty print.
* **m** (int, optional) : The maximum size of the pretty printer. Defaults to the terminal size.


**Returns**

* **function**  : A function that pretty prints values according to the headers.


----


### convert_size
```python
.convert_size(
   size_bytes
)
```

---
Convert a size in bytes to a human-readable string.


**Args**

* **size_bytes** (int) : The size in bytes.


**Returns**

* **str**  : The human-readable string representation of the size.


----


### get_class_that_defined_method
```python
.get_class_that_defined_method(
   meth
)
```

---
Get the class that defined a method.


**Args**

* **meth** (function) : The method to get the defining class of.


**Returns**

* **type**  : The class that defined the method, or None if the method is not a method of a class.


----


### get_module_functions
```python
.get_module_functions(
   module
)
```

---
Get the functions of a module.


**Args**

* **module** (module) : The module to get the functions of.


**Returns**

* **list**  : A list of tuples, where each tuple contains the name of a function and the function itself.


----


### get_module_classes
```python
.get_module_classes(
   module
)
```

---
Get the classes of a module.


**Args**

* **module** (module) : The module to get the classes of.


**Returns**

* **list**  : A list of tuples, where each tuple contains the name of a class and the class itself.

