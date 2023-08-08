#


## POINT
```python 
POINT()
```



----


## RECT
```python 
RECT()
```



----


## RectangleException
```python 
RectangleException()
```



----


## GetWindowException
```python 
GetWindowException()
```



----


## Rectangle
```python 
Rectangle(
   left = 0, top = 0, width = 0, height = 0, enable_float = False, read_only = False,
   on_change = None, on_read = None
)
```




**Methods:**


### .call_on_change
```python
.call_on_change(
   old_left, old_top, old_width, old_height
)
```


### .enable_float
```python
.enable_float()
```


### .left
```python
.left()
```


### .top
```python
.top()
```


### .right
```python
.right()
```


### .bottom
```python
.bottom()
```


### .top_left
```python
.top_left()
```


### .bottom_left
```python
.bottom_left()
```


### .top_right
```python
.top_right()
```


### .bottom_right
```python
.bottom_right()
```


### .mid_top
```python
.mid_top()
```


### .mid_bottom
```python
.mid_bottom()
```


### .mid_left
```python
.mid_left()
```


### .mid_right
```python
.mid_right()
```


### .center
```python
.center()
```


### .center_x
```python
.center_x()
```


### .center_y
```python
.center_y()
```


### .size
```python
.size()
```


### .width
```python
.width()
```


### .height
```python
.height()
```


### .area
```python
.area()
```


### .box
```python
.box()
```


### .get
```python
.get(
   rect_attr_name
)
```


### .set
```python
.set(
   rect_attr_name, value
)
```


### .move
```python
.move(
   x_offset, y_offset
)
```


### .copy
```python
.copy()
```


### .inflate
```python
.inflate(
   width_change = 0, height_change = 0
)
```


### .clamp
```python
.clamp(
   other_rect
)
```


### .union
```python
.union(
   other_rect
)
```


### .union_all
```python
.union_all(
   other_rectangles
)
```


### .normalize
```python
.normalize()
```


### .collide
```python
.collide(
   value
)
```


----


## Window
```python 
Window(
   h_wnd
)
```




**Methods:**


### .close
```python
.close()
```


### .minimize
```python
.minimize()
```


### .maximize
```python
.maximize()
```


### .restore
```python
.restore()
```


### .activate
```python
.activate()
```


### .resize_rel
```python
.resize_rel(
   width_offset, height_offset
)
```


### .resize_to
```python
.resize_to(
   new_width, new_height
)
```


### .move_rel
```python
.move_rel(
   x_offset, y_offset
)
```


### .move_to
```python
.move_to(
   new_left, new_top
)
```


### .h_wnd
```python
.h_wnd()
```


### .is_minimized
```python
.is_minimized()
```


### .is_maximized
```python
.is_maximized()
```


### .is_active
```python
.is_active()
```


### .title
```python
.title()
```


### .visible
```python
.visible()
```


### .left
```python
.left()
```


### .right
```python
.right()
```


### .top
```python
.top()
```


### .bottom
```python
.bottom()
```


### .top_left
```python
.top_left()
```


### .top_right
```python
.top_right()
```


### .bottom_left
```python
.bottom_left()
```


### .bottom_right
```python
.bottom_right()
```


### .mid_left
```python
.mid_left()
```


### .mid_right
```python
.mid_right()
```


### .mid_top
```python
.mid_top()
```


### .mid_bottom
```python
.mid_bottom()
```


### .center
```python
.center()
```


### .center_x
```python
.center_x()
```


### .center_y
```python
.center_y()
```


### .width
```python
.width()
```


### .height
```python
.height()
```


### .size
```python
.size()
```


### .area
```python
.area()
```


### .box
```python
.box()
```


----


### _check_for_int_or_float
```python
._check_for_int_or_float(
   arg
)
```


----


### _check_for_two_int_or_float_tuple
```python
._check_for_two_int_or_float_tuple(
   arg
)
```


----


### _check_for_four_int_or_float_tuple
```python
._check_for_four_int_or_float_tuple(
   arg
)
```


----


### _format_message
```python
._format_message(
   error_code
)
```


----


### _raise_with_last_error
```python
._raise_with_last_error()
```


----


### point_in_rect
```python
.point_in_rect(
   x, y, left, top, width, height
)
```


----


### _get_all_titles
```python
._get_all_titles()
```


----


### get_windows_at
```python
.get_windows_at(
   x, y
)
```


----


### get_windows_with_title
```python
.get_windows_with_title(
   title
)
```


----


### get_all_titles
```python
.get_all_titles()
```


----


### get_all_windows
```python
.get_all_windows()
```


----


### get_window_rect
```python
.get_window_rect(
   h_wnd
)
```


----


### get_active_window
```python
.get_active_window()
```


----


### get_window_text
```python
.get_window_text(
   h_wnd
)
```

