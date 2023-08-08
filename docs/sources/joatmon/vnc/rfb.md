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


----


## BufferReader
```python 
BufferReader(
   buffer: bytes
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


## BufferWriter
```python 
BufferWriter(
   buffer = b''
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


### .get_buffer
```python
.get_buffer()
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


## BaseDES
```python 
BaseDES(
   mode = ECB, iv = None, pad = None, padmode = PAD_NORMAL
)
```




**Methods:**


### .get_key
```python
.get_key()
```


### .set_key
```python
.set_key(
   key
)
```


### .get_mode
```python
.get_mode()
```


### .set_mode
```python
.set_mode(
   mode
)
```


### .get_padding
```python
.get_padding()
```


### .set_padding
```python
.set_padding(
   pad
)
```


### .get_pad_mode
```python
.get_pad_mode()
```


### .set_pad_mode
```python
.set_pad_mode(
   mode
)
```


### .get_iv
```python
.get_iv()
```


### .set_iv
```python
.set_iv(
   iv
)
```


### ._guard_against_unicode
```python
._guard_against_unicode(
   data
)
```


----


## DES
```python 
DES(
   key, mode = ECB, iv = None, pad = None, padmode = PAD_NORMAL
)
```




**Methods:**


### .set_key
```python
.set_key(
   key
)
```


### .__string_to_bit_list
```python
.__string_to_bit_list(
   data
)
```


### .__bit_list_to_string
```python
.__bit_list_to_string(
   data
)
```


### .__permutate
```python
.__permutate(
   table, block
)
```


### .crypt
```python
.crypt(
   data, crypt_type
)
```


### .encrypt
```python
.encrypt(
   data, pad = None, padmode = None
)
```


### .decrypt
```python
.decrypt(
   data, pad = None, padmode = None
)
```


----


## TripleDES
```python 
TripleDES(
   key, mode = ECB, iv = None, pad = None, padmode = PAD_NORMAL
)
```




**Methods:**


### .set_key
```python
.set_key(
   key
)
```


### .set_mode
```python
.set_mode(
   mode
)
```


### .set_padding
```python
.set_padding(
   pad
)
```


### .set_pad_mode
```python
.set_pad_mode(
   mode
)
```


### .set_iv
```python
.set_iv(
   iv
)
```


### .encrypt
```python
.encrypt(
   data, pad = None, padmode = None
)
```


### .decrypt
```python
.decrypt(
   data, pad = None, padmode = None
)
```


----


## RFBClient
```python 
RFBClient(
   host, port, loop
)
```




**Methods:**


### .connect
```python
.connect()
```


### .vnc_request_password
```python
.vnc_request_password()
```


### .vnc_connection_made
```python
.vnc_connection_made()
```


### .vnc_auth_failed
```python
.vnc_auth_failed(
   reason
)
```


### .send_password
```python
.send_password(
   password
)
```


### .begin_update
```python
.begin_update()
```


### .commit_update
```python
.commit_update(
   positions
)
```


### .copy_rectangle
```python
.copy_rectangle(
   srcx, srcy, x, y, width, height
)
```


### .update_rectangle
```python
.update_rectangle(
   x, y, width, height, buffer
)
```


### .fill_rectangle
```python
.fill_rectangle(
   x, y, width, height, color
)
```


### .update_cursor
```python
.update_cursor(
   x, y, width, height, image, mask
)
```


### .update_desktop_size
```python
.update_desktop_size(
   width, height
)
```


### .copy_text
```python
.copy_text(
   text
)
```


### .bell
```python
.bell()
```


### .set_pixel_format
```python
.set_pixel_format(
   bpp = 32, depth = 24, bigendian = 0, truecolor = 1, redmax = 255, greenmax = 255,
   bluemax = 255, redshift = 0, greenshift = 8, blueshift = 16
)
```


### .set_encodings
```python
.set_encodings(
   list_of_encodings
)
```


### .framebuffer_update_request
```python
.framebuffer_update_request(
   x = 0, y = 0, width = None, height = None, incremental = 0
)
```


### .key_event
```python
.key_event(
   key, down = 1
)
```


### .pointer_event
```python
.pointer_event(
   x, y, buttonmask = 0
)
```


### .client_cut_text
```python
.client_cut_text(
   message
)
```


----


## RFBDes
```python 
RFBDes()
```




**Methods:**


### .set_key
```python
.set_key(
   key
)
```


----


### _zrle_next_bit
```python
._zrle_next_bit(
   it, pixels_in_tile
)
```


----


### _zrle_next_dibit
```python
._zrle_next_dibit(
   it, pixels_in_tile
)
```


----


### _zrle_next_nibble
```python
._zrle_next_nibble(
   it, pixels_in_tile
)
```

