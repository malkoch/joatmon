#


## AuthenticationError
```python 
AuthenticationError()
```



----


## VNCClient
```python 
VNCClient(
   host, port, loop
)
```




**Methods:**


### .pause
```python
.pause(
   duration
)
```


### .key_press
```python
.key_press(
   key
)
```


### .key_down
```python
.key_down(
   key
)
```


### .key_up
```python
.key_up(
   key
)
```


### .mouse_press
```python
.mouse_press(
   button
)
```


### .mouse_down
```python
.mouse_down(
   button
)
```


### .mouse_up
```python
.mouse_up(
   button
)
```


### .capture_screen
```python
.capture_screen(
   filename, incremental = 0
)
```


### .capture_region
```python
.capture_region(
   filename, x, y, w, h, incremental = 0
)
```


### .refresh_screen
```python
.refresh_screen(
   incremental = 0
)
```


### .expect_screen
```python
.expect_screen(
   filename, maxrms = 0
)
```


### .expect_region
```python
.expect_region(
   filename, x, y, maxrms = 0
)
```


### .mouse_move
```python
.mouse_move(
   x, y
)
```


### .mouse_drag
```python
.mouse_drag(
   x, y, step = 1
)
```


### .set_image_mode
```python
.set_image_mode()
```


### .vnc_connection_made
```python
.vnc_connection_made()
```


### .vnc_request_password
```python
.vnc_request_password()
```


### .vnc_auth_failed
```python
.vnc_auth_failed(
   reason
)
```


### .begin_update
```python
.begin_update()
```


### .commit_update
```python
.commit_update(
   rectangles = None
)
```


### .update_rectangle
```python
.update_rectangle(
   x, y, width, height, data
)
```


### .copy_rectangle
```python
.copy_rectangle(
   srcx, srcy, x, y, width, height
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


### .draw_cursor
```python
.draw_cursor()
```


### .update_desktop_size
```python
.update_desktop_size(
   width, height
)
```


### .bell
```python
.bell()
```


### .copy_text
```python
.copy_text(
   text
)
```


### .paste
```python
.paste(
   message
)
```

