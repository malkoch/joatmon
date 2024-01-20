#


### producer
```python
.producer(
   plugin, topic
)
```

---
Decorator for handling exceptions in a function.

This decorator wraps the function in a try-except block. If the function raises an exception of type `ex`, the exception is caught and its message is printed. The function then returns None.


**Args**

* **ex** (Exception, optional) : The type of exception to catch. If None, all exceptions are caught. Defaults to None.


**Returns**

* **function**  : The decorated function.


----


### loop
```python
.loop(
   topic, cons
)
```

---
Function for consuming messages from a topic in a loop.

This function retrieves a consumer from the context and uses it to consume messages from a specified topic in a loop. When a message is consumed, it is printed and an event is fired with the arguments and keyword arguments from the message.


**Args**

* **topic** (str) : The topic to consume messages from.
* **cons** (Consumer) : The consumer to use.


----


### consumer_loop_creator
```python
.consumer_loop_creator()
```

---
Function for creating consumer loops.

This function creates a consumer loop for each consumer in the context. If a consumer loop for a consumer already exists, it is not created again.

----


### add_consumer
```python
.add_consumer(
   topic, c
)
```

---
Function for adding a consumer to the context.

This function adds a consumer to the context and creates an event for it.


**Args**

* **topic** (str) : The topic the consumer consumes messages from.
* **c** (Consumer) : The consumer to add.


----


### consumer
```python
.consumer(
   plugin, topic
)
```

---
Decorator for consuming messages from a topic.

This decorator retrieves a consumer from the context and uses it to consume messages from a specified topic. When a message is consumed, an event is fired with the arguments and keyword arguments from the message.


**Args**

* **plugin** (str) : The name of the plugin in the context.
* **topic** (str) : The topic to consume messages from.


**Returns**

* **function**  : The decorated function.

