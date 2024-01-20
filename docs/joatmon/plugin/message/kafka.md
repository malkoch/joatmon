#


## KafkaProducer
```python 
KafkaProducer(
   producer: confluent_kafka.Producer
)
```


---
KafkaProducer class that inherits from the Producer class. It implements the produce method using Kafka.


**Attributes**

* **producer** (confluent_kafka.Producer) : The Kafka producer instance.



**Methods:**


### .produce
```python
.produce(
   topic: str, message: str
)
```

---
Sends a message to a specified Kafka topic.


**Args**

* **topic** (str) : The topic to which the message should be sent.
* **message** (str) : The message to be sent.


----


## KafkaConsumer
```python 
KafkaConsumer(
   consumer: confluent_kafka.Consumer
)
```


---
KafkaConsumer class that inherits from the Consumer class. It implements the consume method using Kafka.


**Attributes**

* **consumer** (confluent_kafka.Consumer) : The Kafka consumer instance.



**Methods:**


### .consume
```python
.consume()
```

---
Receives a message from a Kafka topic.


**Returns**

* **str**  : The received message.


----


## KafkaPlugin
```python 
KafkaPlugin(
   host
)
```


---
KafkaPlugin class that inherits from the MessagePlugin class. It provides the functionality for producing and consuming messages using Kafka.


**Attributes**

* **conf** (dict) : The configuration for Kafka.



**Methods:**


### .get_producer
```python
.get_producer(
   topic
)
```

---
Returns a KafkaProducer for a specified topic.


**Args**

* **topic** (str) : The topic for which a KafkaProducer should be returned.


**Returns**

* **KafkaProducer**  : The KafkaProducer for the specified topic.


### .get_consumer
```python
.get_consumer(
   topic
)
```

---
Returns a KafkaConsumer for a specified topic.


**Args**

* **topic** (str) : The topic for which a KafkaConsumer should be returned.


**Returns**

* **KafkaConsumer**  : The KafkaConsumer for the specified topic.

