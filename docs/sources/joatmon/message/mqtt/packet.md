#


## MQTTFixedHeader
```python 
MQTTFixedHeader(
   packet_type, flags = 0, length = 0
)
```




**Methods:**


### .to_bytes
```python
.to_bytes()
```


### .to_stream
```python
.to_stream(
   writer: WriterAdapter
)
```


### .bytes_length
```python
.bytes_length()
```


### .from_stream
```python
.from_stream(
   cls, reader: ReaderAdapter
)
```


----


## MQTTVariableHeader
```python 

```




**Methods:**


### .to_stream
```python
.to_stream(
   writer: asyncio.StreamWriter
)
```


### .to_bytes
```python
.to_bytes()
```


### .bytes_length
```python
.bytes_length()
```


### .from_stream
```python
.from_stream(
   cls, reader: asyncio.StreamReader, fixed_header: MQTTFixedHeader
)
```


----


## PacketIdVariableHeader
```python 
PacketIdVariableHeader(
   packet_id
)
```




**Methods:**


### .to_bytes
```python
.to_bytes()
```


### .from_stream
```python
.from_stream(
   cls, reader: ReaderAdapter, fixed_header: MQTTFixedHeader
)
```


----


## MQTTPayload
```python 

```




**Methods:**


### .to_stream
```python
.to_stream(
   writer: asyncio.StreamWriter
)
```


### .to_bytes
```python
.to_bytes(
   fixed_header: MQTTFixedHeader, variable_header: MQTTVariableHeader
)
```


### .from_stream
```python
.from_stream(
   cls, reader: asyncio.StreamReader, fixed_header: MQTTFixedHeader,
   variable_header: MQTTVariableHeader
)
```


----


## MQTTPacket
```python 
MQTTPacket(
   fixed: MQTTFixedHeader, variable_header: MQTTVariableHeader = None,
   payload: MQTTPayload = None
)
```




**Methods:**


### .to_stream
```python
.to_stream(
   writer: asyncio.StreamWriter
)
```


### .to_bytes
```python
.to_bytes()
```


### .from_stream
```python
.from_stream(
   cls, reader: ReaderAdapter, fixed_header = None, variable_header = None
)
```


### .bytes_length
```python
.bytes_length()
```


----


## ConnackVariableHeader
```python 
ConnackVariableHeader(
   session_parent = None, return_code = None
)
```




**Methods:**


### .from_stream
```python
.from_stream(
   cls, reader: ReaderAdapter, fixed_header: MQTTFixedHeader
)
```


### .to_bytes
```python
.to_bytes()
```


----


## ConnackPacket
```python 
ConnackPacket(
   fixed: MQTTFixedHeader = None, variable_header: ConnackVariableHeader = None,
   payload = None
)
```




**Methods:**


### .return_code
```python
.return_code()
```


### .session_parent
```python
.session_parent()
```


### .build
```python
.build(
   cls, session_parent = None, return_code = None
)
```


----


## ConnectVariableHeader
```python 
ConnectVariableHeader(
   connect_flags = 0, keep_alive = 0, proto_name = 'MQTT', proto_level = 4
)
```




**Methods:**


### .username_flag
```python
.username_flag()
```


### .password_flag
```python
.password_flag()
```


### .will_retain_flag
```python
.will_retain_flag()
```


### .will_flag
```python
.will_flag()
```


### .clean_session_flag
```python
.clean_session_flag()
```


### .reserved_flag
```python
.reserved_flag()
```


### .will_qos
```python
.will_qos()
```


### .from_stream
```python
.from_stream(
   cls, reader: ReaderAdapter, fixed_header: MQTTFixedHeader
)
```


### .to_bytes
```python
.to_bytes()
```


----


## ConnectPayload
```python 
ConnectPayload(
   client_id = None, will_topic = None, will_message = None, username = None,
   password = None
)
```




**Methods:**


### .from_stream
```python
.from_stream(
   cls, reader: ReaderAdapter, fixed_header: MQTTFixedHeader,
   variable_header: ConnectVariableHeader
)
```


### .to_bytes
```python
.to_bytes(
   fixed_header: MQTTFixedHeader, variable_header: ConnectVariableHeader
)
```


----


## ConnectPacket
```python 
ConnectPacket(
   fixed: MQTTFixedHeader = None, vh: ConnectVariableHeader = None,
   payload: ConnectPayload = None
)
```




**Methods:**


### .proto_name
```python
.proto_name()
```


### .proto_level
```python
.proto_level()
```


### .username_flag
```python
.username_flag()
```


### .password_flag
```python
.password_flag()
```


### .clean_session_flag
```python
.clean_session_flag()
```


### .will_retain_flag
```python
.will_retain_flag()
```


### .will_qos
```python
.will_qos()
```


### .will_flag
```python
.will_flag()
```


### .reserved_flag
```python
.reserved_flag()
```


### .client_id
```python
.client_id()
```


### .client_id_is_random
```python
.client_id_is_random()
```


### .will_topic
```python
.will_topic()
```


### .will_message
```python
.will_message()
```


### .username
```python
.username()
```


### .password
```python
.password()
```


### .keep_alive
```python
.keep_alive()
```


----


## DisconnectPacket
```python 
DisconnectPacket(
   fixed: MQTTFixedHeader = None
)
```



----


## PingReqPacket
```python 
PingReqPacket(
   fixed: MQTTFixedHeader = None
)
```



----


## PingRespPacket
```python 
PingRespPacket(
   fixed: MQTTFixedHeader = None
)
```




**Methods:**


### .build
```python
.build(
   cls
)
```


----


## PubackPacket
```python 
PubackPacket(
   fixed: MQTTFixedHeader = None, variable_header: PacketIdVariableHeader = None
)
```




**Methods:**


### .packet_id
```python
.packet_id()
```


### .build
```python
.build(
   cls, packet_id: int
)
```


----


## PubcompPacket
```python 
PubcompPacket(
   fixed: MQTTFixedHeader = None, variable_header: PacketIdVariableHeader = None
)
```




**Methods:**


### .packet_id
```python
.packet_id()
```


### .build
```python
.build(
   cls, packet_id: int
)
```


----


## PublishVariableHeader
```python 
PublishVariableHeader(
   topic_name: str, packet_id: int = None
)
```




**Methods:**


### .to_bytes
```python
.to_bytes()
```


### .from_stream
```python
.from_stream(
   cls, reader: asyncio.StreamReader, fixed_header: MQTTFixedHeader
)
```


----


## PublishPayload
```python 
PublishPayload(
   data: bytes = None
)
```




**Methods:**


### .to_bytes
```python
.to_bytes(
   fixed_header: MQTTFixedHeader, variable_header: MQTTVariableHeader
)
```


### .from_stream
```python
.from_stream(
   cls, reader: asyncio.StreamReader, fixed_header: MQTTFixedHeader,
   variable_header: MQTTVariableHeader
)
```


----


## PublishPacket
```python 
PublishPacket(
   fixed: MQTTFixedHeader = None, variable_header: PublishVariableHeader = None,
   payload = None
)
```




**Methods:**


### .set_flags
```python
.set_flags(
   dup_flag = False, qos = 0, retain_flag = False
)
```


### .dup_flag
```python
.dup_flag()
```


### .retain_flag
```python
.retain_flag()
```


### .qos
```python
.qos()
```


### .packet_id
```python
.packet_id()
```


### .data
```python
.data()
```


### .topic_name
```python
.topic_name()
```


### .build
```python
.build(
   cls, topic_name: str, message: bytes, packet_id: int, dup_flag, qos, retain
)
```


----


## PubrecPacket
```python 
PubrecPacket(
   fixed: MQTTFixedHeader = None, variable_header: PacketIdVariableHeader = None
)
```




**Methods:**


### .packet_id
```python
.packet_id()
```


### .build
```python
.build(
   cls, packet_id: int
)
```


----


## PubrelPacket
```python 
PubrelPacket(
   fixed: MQTTFixedHeader = None, variable_header: PacketIdVariableHeader = None
)
```




**Methods:**


### .packet_id
```python
.packet_id()
```


### .build
```python
.build(
   cls, packet_id
)
```


----


## SubackPayload
```python 
SubackPayload(
   return_codes = []
)
```




**Methods:**


### .to_bytes
```python
.to_bytes(
   fixed_header: MQTTFixedHeader, variable_header: MQTTVariableHeader
)
```


### .from_stream
```python
.from_stream(
   cls, reader: ReaderAdapter, fixed_header: MQTTFixedHeader,
   variable_header: MQTTVariableHeader
)
```


----


## SubackPacket
```python 
SubackPacket(
   fixed: MQTTFixedHeader = None, variable_header: PacketIdVariableHeader = None,
   payload = None
)
```




**Methods:**


### .build
```python
.build(
   cls, packet_id, return_codes
)
```


----


## SubscribePayload
```python 
SubscribePayload(
   topics = []
)
```




**Methods:**


### .to_bytes
```python
.to_bytes(
   fixed_header: MQTTFixedHeader, variable_header: MQTTVariableHeader
)
```


### .from_stream
```python
.from_stream(
   cls, reader: asyncio.StreamReader, fixed_header: MQTTFixedHeader,
   variable_header: MQTTVariableHeader
)
```


----


## SubscribePacket
```python 
SubscribePacket(
   fixed: MQTTFixedHeader = None, variable_header: PacketIdVariableHeader = None,
   payload = None
)
```




**Methods:**


### .build
```python
.build(
   cls, topics, packet_id
)
```


----


## UnsubackPacket
```python 
UnsubackPacket(
   fixed: MQTTFixedHeader = None, variable_header: PacketIdVariableHeader = None,
   payload = None
)
```




**Methods:**


### .build
```python
.build(
   cls, packet_id
)
```


----


## UnubscribePayload
```python 
UnubscribePayload(
   topics = []
)
```




**Methods:**


### .to_bytes
```python
.to_bytes(
   fixed_header: MQTTFixedHeader, variable_header: MQTTVariableHeader
)
```


### .from_stream
```python
.from_stream(
   cls, reader: asyncio.StreamReader, fixed_header: MQTTFixedHeader,
   variable_header: MQTTVariableHeader
)
```


----


## UnsubscribePacket
```python 
UnsubscribePacket(
   fixed: MQTTFixedHeader = None, variable_header: PacketIdVariableHeader = None,
   payload = None
)
```




**Methods:**


### .build
```python
.build(
   cls, topics, packet_id
)
```

