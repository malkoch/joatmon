#


## ApplicationMessage
```python 
ApplicationMessage(
   packet_id, topic, qos, data, retain
)
```




**Methods:**


### .build_publish_packet
```python
.build_publish_packet(
   dup = False
)
```


----


## IncomingApplicationMessage
```python 
IncomingApplicationMessage(
   packet_id, topic, qos, data, retain
)
```



----


## OutgoingApplicationMessage
```python 
OutgoingApplicationMessage(
   packet_id, topic, qos, data, retain
)
```



----


## Session
```python 
Session(
   loop = None
)
```




**Methods:**


### .next_packet_id
```python
.next_packet_id()
```


### .inflight_in_count
```python
.inflight_in_count()
```


### .inflight_out_count
```python
.inflight_out_count()
```


### .retained_messages_count
```python
.retained_messages_count()
```

