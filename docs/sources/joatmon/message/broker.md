#


## BrokerException
```python 
BrokerException()
```



----


## RetainedApplicationMessage
```python 
RetainedApplicationMessage(
   source_session, topic, data, qos = None
)
```



----


## Server
```python 
Server(
   listener_name, server_instance, max_connections = -1, loop = None
)
```




**Methods:**


### .acquire_connection
```python
.acquire_connection()
```


### .release_connection
```python
.release_connection()
```


### .close_instance
```python
.close_instance()
```


----


## Broker
```python 
Broker(
   config = None, loop = None
)
```




**Methods:**


### .start
```python
.start()
```


### .shutdown
```python
.shutdown()
```


### .internal_message_broadcast
```python
.internal_message_broadcast(
   topic, data, qos = None
)
```


### .ws_connected
```python
.ws_connected(
   websocket, uri, listener_name
)
```


### .stream_connected
```python
.stream_connected(
   reader, writer, listener_name
)
```


### .client_connected
```python
.client_connected(
   listener_name, reader: ReaderAdapter, writer: WriterAdapter
)
```


### .authenticate
```python
.authenticate(
   session: Session, listener
)
```


### .topic_filtering
```python
.topic_filtering(
   session: Session, topic
)
```


### .retain_message
```python
.retain_message(
   source_session, topic_name, data, qos = None
)
```


### .add_subscription
```python
.add_subscription(
   subscription, session
)
```


### .matches
```python
.matches(
   topic, a_filter
)
```


### .publish_session_retained_messages
```python
.publish_session_retained_messages(
   session
)
```


### .publish_retained_messages_for_subscription
```python
.publish_retained_messages_for_subscription(
   subscription, session
)
```


### .delete_session
```python
.delete_session(
   client_id
)
```

