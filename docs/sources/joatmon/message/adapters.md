#


## ReaderAdapter
```python 
ReaderAdapter()
```




**Methods:**


### .read
```python
.read(
   n = -1
)
```


### .feed_eof
```python
.feed_eof()
```


----


## WriterAdapter
```python 
WriterAdapter()
```




**Methods:**


### .write
```python
.write(
   data
)
```


### .drain
```python
.drain()
```


### .get_peer_info
```python
.get_peer_info()
```


### .close
```python
.close()
```


----


## WebSocketsReader
```python 
WebSocketsReader(
   protocol: WebSocketCommonProtocol
)
```




**Methods:**


### .read
```python
.read(
   n = -1
)
```


----


## WebSocketsWriter
```python 
WebSocketsWriter(
   protocol: WebSocketCommonProtocol
)
```




**Methods:**


### .write
```python
.write(
   data
)
```


### .drain
```python
.drain()
```


### .get_peer_info
```python
.get_peer_info()
```


### .close
```python
.close()
```


----


## StreamReaderAdapter
```python 
StreamReaderAdapter(
   reader: StreamReader
)
```




**Methods:**


### .read
```python
.read(
   n = -1
)
```


### .feed_eof
```python
.feed_eof()
```


----


## StreamWriterAdapter
```python 
StreamWriterAdapter(
   writer: StreamWriter
)
```




**Methods:**


### .write
```python
.write(
   data
)
```


### .drain
```python
.drain()
```


### .get_peer_info
```python
.get_peer_info()
```


### .close
```python
.close()
```

