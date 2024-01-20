#


## RedisCache
```python 
RedisCache(
   host: str, port: int, password: str
)
```


---
RedisCache class that inherits from the CachePlugin class. It implements the abstract methods of the CachePlugin class
using Redis for caching.


**Attributes**

* **host** (str) : The host of the Redis server.
* **port** (int) : The port of the Redis server.
* **password** (str) : The password for the Redis server.
* **connection** (`redis.Redis` instance) : The connection to the Redis server.



**Methods:**


### .add
```python
.add(
   key, value, duration = None
)
```

---
Add a key-value pair to the cache with an optional duration.


**Args**

* **key** (str) : The key of the item to be added.
* **value** (str) : The value of the item to be added.
* **duration** (int, optional) : The duration for which the item should be stored in the cache.


### .get
```python
.get(
   key
)
```

---
Retrieve a value from the cache using its key.


**Args**

* **key** (str) : The key of the item to be retrieved.


**Returns**

* **str**  : The value of the item.


### .update
```python
.update(
   key, value
)
```

---
Update the value of an item in the cache using its key.


**Args**

* **key** (str) : The key of the item to be updated.
* **value** (str) : The new value of the item.


### .remove
```python
.remove(
   key
)
```

---
Remove an item from the cache using its key.


**Args**

* **key** (str) : The key of the item to be removed.

