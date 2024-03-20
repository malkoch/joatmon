#


## RWLock
```python 
RWLock(
   modes = None, max_read = 0
)
```


---
A class used to represent a Read-Write Lock.

Attributes
----------
w_lock : Lock
The lock for write operations.
---
    The lock for read operations.
    The number of read operations.

Methods
-------
__init__(self)
    Initializes a new instance of the RWLock class.
r_acquire(self)
    Acquires the read lock.
r_release(self)
    Releases the read lock.
r_locked(self)
    Context manager for read lock.
w_acquire(self)
    Acquires the write lock.
w_release(self)
    Releases the write lock.
w_locked(self)
    Context manager for write lock.


**Methods:**


### .read
```python
.read()
```


### .write
```python
.write()
```


### .r_acquire
```python
.r_acquire()
```

---
Acquires the read lock.

This method increases the number of read operations and acquires the write lock if it's the first read operation.

### .r_release
```python
.r_release()
```

---
Releases the read lock.

This method decreases the number of read operations and releases the write lock if there are no more read operations.

### .r_locked
```python
.r_locked()
```

---
Context manager for read lock.

This method acquires the read lock before entering the context and releases it after exiting the context.

### .w_acquire
```python
.w_acquire()
```

---
Acquires the write lock.

### .w_release
```python
.w_release()
```

---
Releases the write lock.

### .w_locked
```python
.w_locked()
```

---
Context manager for write lock.

This method acquires the write lock before entering the context and releases it after exiting the context.
