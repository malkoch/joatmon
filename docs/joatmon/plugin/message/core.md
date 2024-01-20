#


## Producer
```python 
Producer()
```


---
Abstract Producer class that defines the interface for producing messages.

Methods:
produce: Sends a message to a specified topic.


**Methods:**


### .produce
```python
.produce(
   topic: str, message: str
)
```

---
Sends a message to a specified topic.


**Args**

* **topic** (str) : The topic to which the message should be sent.
* **message** (str) : The message to be sent.


**Raises**

* **NotImplementedError**  : This method should be implemented in the child classes.


----


## Consumer
```python 
Consumer()
```


---
Abstract Consumer class that defines the interface for consuming messages.

Methods:
consume: Receives a message.


**Methods:**


### .consume
```python
.consume()
```

---
Receives a message.


**Returns**

* **str**  : The received message.


**Raises**

* **NotImplementedError**  : This method should be implemented in the child classes.


----


## MessagePlugin
```python 
MessagePlugin()
```


---
MessagePlugin class that inherits from the Plugin class. It provides the functionality for producing and consuming messages.

Methods:
get_producer: Returns a Producer for a specified topic.
get_consumer: Returns a Consumer for a specified topic.


**Methods:**


### .get_producer
```python
.get_producer(
   topic
)
```

---
Returns a Producer for a specified topic.


**Args**

* **topic** (str) : The topic for which a Producer should be returned.


**Returns**

* **Producer**  : The Producer for the specified topic.


**Raises**

* **NotImplementedError**  : This method should be implemented in the child classes.


### .get_consumer
```python
.get_consumer(
   topic
)
```

---
Returns a Consumer for a specified topic.


**Args**

* **topic** (str) : The topic for which a Consumer should be returned.


**Returns**

* **Consumer**  : The Consumer for the specified topic.


**Raises**

* **NotImplementedError**  : This method should be implemented in the child classes.

