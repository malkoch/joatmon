#


## ClientException
```python 
ClientException()
```



----


## ConnectException
```python 
ConnectException()
```



----


## MQTTClient
```python 
MQTTClient(
   client_id = None, config = None, loop = None
)
```




**Methods:**


### .connect
```python
.connect(
   uri = None, cleansession = None, cafile = None, capath = None, cadata = None,
   extra_headers = {}
)
```


### .disconnect
```python
.disconnect()
```


### .cancel_tasks
```python
.cancel_tasks()
```


### .reconnect
```python
.reconnect(
   cleansession = None
)
```


### .ping
```python
.ping()
```


### .publish
```python
.publish(
   topic, message, qos = None, retain = None, ack_timeout = None
)
```


### .subscribe
```python
.subscribe(
   topics
)
```


### .unsubscribe
```python
.unsubscribe(
   topics
)
```


### .deliver_message
```python
.deliver_message(
   timeout = None
)
```


### .handle_connection_close
```python
.handle_connection_close()
```


----


### mqtt_connected
```python
.mqtt_connected(
   func
)
```

