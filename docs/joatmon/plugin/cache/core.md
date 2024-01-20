#


## CachePlugin
```python 
CachePlugin()
```


---
CachePlugin class that inherits from the Plugin class. It is an abstract class that provides
the structure for caching methods. The methods in this class should be implemented in the child classes.


**Methods:**


### .add
```python
.add(
   key, value, duration = None
)
```

---
This is an abstract method that should be implemented in the child classes. It is used to
add a key-value pair to the cache.


**Args**

* **key** (str) : The key of the item to be added.
* **value** (str) : The value of the item to be added.
* **duration** (int, optional) : The duration for which the item should be stored in the cache.


**Raises**

* **NotImplementedError**  : This method should be implemented in the child classes.


### .get
```python
.get(
   key
)
```

---
This is an abstract method that should be implemented in the child classes. It is used to
retrieve a value from the cache using its key.


**Args**

* **key** (str) : The key of the item to be retrieved.


**Raises**

* **NotImplementedError**  : This method should be implemented in the child classes.


### .update
```python
.update(
   key, value
)
```

---
This is an abstract method that should be implemented in the child classes. It is used to
update the value of an item in the cache using its key.


**Args**

* **key** (str) : The key of the item to be updated.
* **value** (str) : The new value of the item.


**Raises**

* **NotImplementedError**  : This method should be implemented in the child classes.


### .remove
```python
.remove(
   key
)
```

---
This is an abstract method that should be implemented in the child classes. It is used to
remove an item from the cache using its key.


**Args**

* **key** (str) : The key of the item to be removed.


**Raises**

* **NotImplementedError**  : This method should be implemented in the child classes.

