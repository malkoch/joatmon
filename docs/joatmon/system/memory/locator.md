#


## Locator
```python 
Locator(
   worker, of_type = 'unknown', start = None, end = None
)
```


---
A class used to represent a Locator.

Attributes
----------
worker : object
The worker object that the locator belongs to.
---
    The type of the locator.
    The last iteration of the locator.
    The last value of the locator.
    The start of the locator.
    The end of the locator.

Methods
-------
__init__(self, worker, of_type='unknown', start=None, end=None)
    Initializes a new instance of the Locator class.
find(self, value, erase_last=True)
    Finds the given value in the locator.
feed(self, value, erase_last=True)
    Feeds the given value into the locator.
get_addresses(self)
    Gets the addresses of the locator.
diff(self, erase_last=False)
    Gets the difference of the locator.
get_modified_address(self, erase_last=False)
    Gets the modified address of the locator.


**Methods:**


### .find
```python
.find(
   value, erase_last = True
)
```

---
Finds the given value in the locator.


**Args**

* **value** (object) : The value to find.
* **erase_last** (bool, optional) : Whether to erase the last value. Defaults to True.


**Returns**

* **dict**  : The new iteration of the locator.


### .feed
```python
.feed(
   value, erase_last = True
)
```

---
Feeds the given value into the locator.


**Args**

* **value** (object) : The value to feed.
* **erase_last** (bool, optional) : Whether to erase the last value. Defaults to True.


**Returns**

* **dict**  : The new iteration of the locator.


### .get_addresses
```python
.get_addresses()
```

---
Gets the addresses of the locator.


**Returns**

* **dict**  : The addresses of the locator.


### .diff
```python
.diff(
   erase_last = False
)
```

---
Gets the difference of the locator.


**Args**

* **erase_last** (bool, optional) : Whether to erase the last value. Defaults to False.


**Returns**

* **dict**  : The difference of the locator.


### .get_modified_address
```python
.get_modified_address(
   erase_last = False
)
```

---
Gets the modified address of the locator.


**Args**

* **erase_last** (bool, optional) : Whether to erase the last value. Defaults to False.


**Returns**

* **dict**  : The modified address of the locator.

