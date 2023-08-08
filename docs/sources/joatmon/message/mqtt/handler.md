#


## ProtocolHandlerException
```python 
ProtocolHandlerException()
```



----


## ProtocolHandler
```python 
ProtocolHandler(
   session: Session = None, loop = None
)
```




**Methods:**


### .attach
```python
.attach(
   session, reader: ReaderAdapter, writer: WriterAdapter
)
```


### .detach
```python
.detach()
```


### .start
```python
.start()
```


### .stop
```python
.stop()
```


### .mqtt_publish
```python
.mqtt_publish(
   topic, data, qos, retain, ack_timeout = None
)
```


### .mqtt_deliver_next_message
```python
.mqtt_deliver_next_message()
```


### .handle_write_timeout
```python
.handle_write_timeout()
```


### .handle_read_timeout
```python
.handle_read_timeout()
```


### .handle_connack
```python
.handle_connack(
   connack: ConnackPacket
)
```


### .handle_connect
```python
.handle_connect(
   connect: ConnectPacket
)
```


### .handle_subscribe
```python
.handle_subscribe(
   subscribe: SubscribePacket
)
```


### .handle_unsubscribe
```python
.handle_unsubscribe(
   subscribe: UnsubscribePacket
)
```


### .handle_suback
```python
.handle_suback(
   suback: SubackPacket
)
```


### .handle_unsuback
```python
.handle_unsuback(
   unsuback: UnsubackPacket
)
```


### .handle_pingresp
```python
.handle_pingresp(
   pingresp: PingRespPacket
)
```


### .handle_pingreq
```python
.handle_pingreq(
   pingreq: PingReqPacket
)
```


### .handle_disconnect
```python
.handle_disconnect(
   disconnect: DisconnectPacket
)
```


### .handle_connection_closed
```python
.handle_connection_closed()
```


### .handle_puback
```python
.handle_puback(
   puback: PubackPacket
)
```


### .handle_pubrec
```python
.handle_pubrec(
   pubrec: PubrecPacket
)
```


### .handle_pubcomp
```python
.handle_pubcomp(
   pubcomp: PubcompPacket
)
```


### .handle_pubrel
```python
.handle_pubrel(
   pubrel: PubrelPacket
)
```


### .handle_publish
```python
.handle_publish(
   publish_packet: PublishPacket
)
```


----


## ClientProtocolHandler
```python 
ClientProtocolHandler(
   session: Session = None, loop = None
)
```




**Methods:**


### .start
```python
.start()
```


### .stop
```python
.stop()
```


### .mqtt_connect
```python
.mqtt_connect()
```


### .handle_write_timeout
```python
.handle_write_timeout()
```


### .handle_read_timeout
```python
.handle_read_timeout()
```


### .mqtt_subscribe
```python
.mqtt_subscribe(
   topics, packet_id
)
```


### .handle_suback
```python
.handle_suback(
   suback: SubackPacket
)
```


### .mqtt_unsubscribe
```python
.mqtt_unsubscribe(
   topics, packet_id
)
```


### .handle_unsuback
```python
.handle_unsuback(
   unsuback: UnsubackPacket
)
```


### .mqtt_disconnect
```python
.mqtt_disconnect()
```


### .mqtt_ping
```python
.mqtt_ping()
```


### .handle_pingresp
```python
.handle_pingresp(
   pingresp: PingRespPacket
)
```


### .handle_connection_closed
```python
.handle_connection_closed()
```


### .wait_disconnect
```python
.wait_disconnect()
```


----


## BrokerProtocolHandler
```python 
BrokerProtocolHandler(
   session: Session = None, loop = None
)
```




**Methods:**


### .start
```python
.start()
```


### .stop
```python
.stop()
```


### .wait_disconnect
```python
.wait_disconnect()
```


### .handle_write_timeout
```python
.handle_write_timeout()
```


### .handle_read_timeout
```python
.handle_read_timeout()
```


### .handle_disconnect
```python
.handle_disconnect(
   disconnect
)
```


### .handle_connection_closed
```python
.handle_connection_closed()
```


### .handle_connect
```python
.handle_connect(
   connect: ConnectPacket
)
```


### .handle_pingreq
```python
.handle_pingreq(
   pingreq: PingReqPacket
)
```


### .handle_subscribe
```python
.handle_subscribe(
   subscribe: SubscribePacket
)
```


### .handle_unsubscribe
```python
.handle_unsubscribe(
   unsubscribe: UnsubscribePacket
)
```


### .get_next_pending_subscription
```python
.get_next_pending_subscription()
```


### .get_next_pending_unsubscription
```python
.get_next_pending_unsubscription()
```


### .mqtt_acknowledge_subscription
```python
.mqtt_acknowledge_subscription(
   packet_id, return_codes
)
```


### .mqtt_acknowledge_unsubscription
```python
.mqtt_acknowledge_unsubscription(
   packet_id
)
```


### .mqtt_connack_authorize
```python
.mqtt_connack_authorize(
   authorize: bool
)
```


### .init_from_connect
```python
.init_from_connect(
   cls, reader: ReaderAdapter, writer: WriterAdapter, loop = None
)
```

