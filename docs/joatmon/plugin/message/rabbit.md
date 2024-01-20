#


## RabbitProducer
```python 
RabbitProducer(
   producer
)
```


---
RabbitProducer class that inherits from the Producer class. It implements the produce method using RabbitMQ.


**Attributes**

* **producer** (pika.BlockingConnection.channel) : The RabbitMQ producer instance.



**Methods:**


### .produce
```python
.produce(
   topic: str, message: str
)
```

---
Sends a message to a specified RabbitMQ topic.


**Args**

* **topic** (str) : The topic to which the message should be sent.
* **message** (str) : The message to be sent.


----


## RabbitConsumer
```python 
RabbitConsumer(
   consumer
)
```


---
RabbitConsumer class that inherits from the Consumer class. It implements the consume method using RabbitMQ.


**Attributes**

* **consumer** (pika.BlockingConnection.channel) : The RabbitMQ consumer instance.



**Methods:**


### .set_q
```python
.set_q(
   q
)
```

---
Set the queue for the consumer.


**Args**

* **q** (queue.Queue) : The queue for the consumer.


### .consume
```python
.consume()
```

---
Receives a message from a RabbitMQ topic.


**Returns**

* **str**  : The received message.


----


## RabbitMQPlugin
```python 
RabbitMQPlugin(
   host, port, username, password
)
```


---
RabbitMQPlugin class that inherits from the MessagePlugin class. It provides the functionality for producing and consuming messages using RabbitMQ.


**Attributes**

* **host** (str) : The host for RabbitMQ.
* **port** (int) : The port for RabbitMQ.
* **username** (str) : The username for RabbitMQ.
* **password** (str) : The password for RabbitMQ.
* **d** (dict) : A dictionary to store the queues for each topic.



**Methods:**


### .get_producer
```python
.get_producer(
   topic
)
```

---
Returns a RabbitProducer for a specified topic.


**Args**

* **topic** (str) : The topic for which a RabbitProducer should be returned.


**Returns**

* **RabbitProducer**  : The RabbitProducer for the specified topic.


### .get_consumer
```python
.get_consumer(
   topic
)
```

---
Returns a RabbitConsumer for a specified topic.


**Args**

* **topic** (str) : The topic for which a RabbitConsumer should be returned.


**Returns**

* **RabbitConsumer**  : The RabbitConsumer for the specified topic.

