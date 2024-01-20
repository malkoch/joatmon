#


## Event
```python 

```


---
Event class for managing event subscribers and firing events.

This class provides a way to manage event subscribers and fire events.


**Attributes**

* **_subscribers** (dict) : The subscribers of the event.



**Methods:**


### .fire
```python
.fire(
   *args, **kwargs
)
```

---
Fire the event to all subscribers.


**Args**

* **args**  : Variable length argument list.
* **kwargs**  : Arbitrary keyword arguments.


----


## AsyncEvent
```python 

```


---
AsyncEvent class for managing asynchronous event subscribers and firing events.

This class provides a way to manage asynchronous event subscribers and fire events.


**Attributes**

* **_subscribers** (dict) : The subscribers of the event.



**Methods:**


### .fire
```python
.fire(
   *args, **kwargs
)
```

---
Fire the event to all subscribers asynchronously.


**Args**

* **args**  : Variable length argument list.
* **kwargs**  : Arbitrary keyword arguments.

