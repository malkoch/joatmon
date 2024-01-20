#


## POINT
```python 
POINT()
```


---
POINT structure for representing a point in a 2D space.

Fields:
x (long): The x-coordinate of the point.
y (long): The y-coordinate of the point.

----


## RECT
```python 
RECT()
```


---
RECT structure for representing a rectangle.

Fields:
left (long): The x-coordinate of the upper-left corner of the rectangle.
top (long): The y-coordinate of the upper-left corner of the rectangle.
right (long): The x-coordinate of the lower-right corner of the rectangle.
bottom (long): The y-coordinate of the lower-right corner of the rectangle.

----


## RectangleException
```python 
RectangleException()
```


---
Exception raised for errors in the Rectangle class.

----


## GetWindowException
```python 
GetWindowException()
```


---
Exception raised for errors in getting window information.

----


## Rectangle
```python 
Rectangle(
   left = 0, top = 0, width = 0, height = 0, enable_float = False, read_only = False,
   on_change = None, on_read = None
)
```


---
Rectangle class for representing and manipulating a rectangle.

Methods:
__init__: Initialize the Rectangle.
__repr__: Return a string representation of the Rectangle.
__str__: Return a string representation of the Rectangle.
call_on_change: Call a function when the Rectangle changes.
enable_float: Enable or disable floating point values.
left: Get or set the left coordinate of the Rectangle.
top: Get or set the top coordinate of the Rectangle.
right: Get or set the right coordinate of the Rectangle.
bottom: Get or set the bottom coordinate of the Rectangle.
top_left: Get or set the top-left coordinate of the Rectangle.
bottom_left: Get or set the bottom-left coordinate of the Rectangle.
top_right: Get or set the top-right coordinate of the Rectangle.
bottom_right: Get or set the bottom-right coordinate of the Rectangle.
mid_top: Get or set the mid-top coordinate of the Rectangle.
mid_bottom: Get or set the mid-bottom coordinate of the Rectangle.
mid_left: Get or set the mid-left coordinate of the Rectangle.
mid_right: Get or set the mid-right coordinate of the Rectangle.
center: Get or set the center coordinate of the Rectangle.
center_x: Get or set the x-coordinate of the center of the Rectangle.
center_y: Get or set the y-coordinate of the center of the Rectangle.
size: Get or set the size of the Rectangle.
width: Get or set the width of the Rectangle.
height: Get or set the height of the Rectangle.
area: Get the area of the Rectangle.
box: Get or set the box of the Rectangle.
get: Get a property of the Rectangle.
set: Set a property of the Rectangle.
move: Move the Rectangle.
copy: Return a copy of the Rectangle.
inflate: Inflate the Rectangle.
clamp: Clamp the Rectangle to another Rectangle.
union: Union the Rectangle with another Rectangle.
union_all: Union the Rectangle with a list of other Rectangles.
normalize: Normalize the Rectangle.
__contains__: Check if a point or another Rectangle is contained in the Rectangle.
collide: Check if a point or another Rectangle collides with the Rectangle.
__eq__: Check if the Rectangle is equal to another Rectangle.
__ne__: Check if the Rectangle is not equal to another Rectangle.


**Methods:**


### .call_on_change
```python
.call_on_change(
   old_left, old_top, old_width, old_height
)
```

---
Calls the onChange function if it is not None.

### .enable_float
```python
.enable_float()
```

---
Returns the value of the _enable_float attribute.

### .left
```python
.left()
```

---
Returns the value of the _left attribute.

### .top
```python
.top()
```

---
Returns the value of the _top attribute.

### .right
```python
.right()
```

---
Returns the sum of the _left and _width attributes.

### .bottom
```python
.bottom()
```

---
Returns the sum of the _top and _height attributes.

### .top_left
```python
.top_left()
```

---
Returns a Point object with the _left and _top attributes as its coordinates.

### .bottom_left
```python
.bottom_left()
```

---
Returns a Point object with the _left and the sum of _top and _height as its coordinates.

### .top_right
```python
.top_right()
```

---
Returns a Point object with the sum of _left and _width and _top as its coordinates.

### .bottom_right
```python
.bottom_right()
```

---
Returns a Point object with the sum of _left and _width and the sum of _top and _height as its coordinates.

### .mid_top
```python
.mid_top()
```

---
Returns a Point object with the middle point of the top edge as its coordinates.

### .mid_bottom
```python
.mid_bottom()
```

---
Returns a Point object with the middle point of the bottom edge as its coordinates.

### .mid_left
```python
.mid_left()
```

---
Returns a Point object with the middle point of the left edge as its coordinates.

### .mid_right
```python
.mid_right()
```

---
Returns a Point object with the middle point of the right edge as its coordinates.

### .center
```python
.center()
```

---
Returns a Point object with the center point of the rectangle as its coordinates.

### .center_x
```python
.center_x()
```

---
Returns the x-coordinate of the center point of the rectangle.

### .center_y
```python
.center_y()
```

---
Returns the y-coordinate of the center point of the rectangle.

### .size
```python
.size()
```

---
Returns a Size object with the _width and _height attributes as its dimensions.

### .width
```python
.width()
```

---
Returns the value of the _width attribute.

### .height
```python
.height()
```

---
Property getter for _height attribute of the Rectangle instance.


**Returns**

* **float**  : The height of the rectangle.


### .area
```python
.area()
```

---
Property getter for the area of the Rectangle instance.


**Returns**

* **float**  : The area of the rectangle.


### .box
```python
.box()
```

---
Property getter for the box representation of the Rectangle instance.


**Returns**

* **Box**  : The box representation of the rectangle.


### .get
```python
.get(
   rect_attr_name
)
```

---
Returns the value of the specified rectangle attribute.


**Args**

* **rect_attr_name** (str) : The name of the rectangle attribute.


**Returns**

* **float**  : The value of the specified rectangle attribute.


**Raises**

* **RectangleException**  : If the attribute name is not valid.


### .set
```python
.set(
   rect_attr_name, value
)
```

---
Sets the value of the specified rectangle attribute.


**Args**

* **rect_attr_name** (str) : The name of the rectangle attribute.
* **value** (int or float) : The new value of the rectangle attribute.


**Raises**

* **RectangleException**  : If the attribute name is not valid or if the attribute is read-only.


### .move
```python
.move(
   x_offset, y_offset
)
```

---
Moves the rectangle by the specified offsets.


**Args**

* **x_offset** (int or float) : The offset along the x-axis.
* **y_offset** (int or float) : The offset along the y-axis.


**Raises**

* **RectangleException**  : If the Rectangle instance is read-only.


### .copy
```python
.copy()
```

---
Returns a copy of the Rectangle instance.


**Returns**

* **Rectangle**  : A copy of the Rectangle instance.


### .inflate
```python
.inflate(
   width_change = 0, height_change = 0
)
```

---
Increases the size of the rectangle by the specified amounts.


**Args**

* **width_change** (int or float) : The change in width. Defaults to 0.
* **height_change** (int or float) : The change in height. Defaults to 0.


**Raises**

* **RectangleException**  : If the Rectangle instance is read-only.


### .clamp
```python
.clamp(
   other_rect
)
```

---
Moves the rectangle to be within the bounds of another rectangle.


**Args**

* **other_rect** (Rectangle) : The other rectangle.


**Raises**

* **RectangleException**  : If the Rectangle instance is read-only.


### .union
```python
.union(
   other_rect
)
```

---
Expands the rectangle to include another rectangle.


**Args**

* **other_rect** (Rectangle) : The other rectangle.


### .union_all
```python
.union_all(
   other_rectangles
)
```

---
Expands the rectangle to include a list of other rectangles.


**Args**

* **other_rectangles** (list) : The list of other rectangles.


### .normalize
```python
.normalize()
```

---
Adjusts the rectangle to have non-negative width and height.


**Raises**

* **RectangleException**  : If the Rectangle instance is read-only.


### .collide
```python
.collide(
   value
)
```

---
Checks whether a point or rectangle collides with the rectangle.


**Args**

* **value** (tuple or Rectangle) : The point or rectangle to check.


**Returns**

* **bool**  : True if the point or rectangle collides with the rectangle, False otherwise.


**Raises**

* **RectangleException**  : If the value is not a valid point or rectangle.


----


## Window
```python 
Window(
   h_wnd
)
```


---
Window class for representing and manipulating a window.

Methods:
__init__: Initialize the Window.
__repr__: Return a string representation of the Window.
__str__: Return a string representation of the Window.
__eq__: Check if the Window is equal to another Window.
close: Close the Window.
minimize: Minimize the Window.
maximize: Maximize the Window.
restore: Restore the Window.
activate: Activate the Window.
resize_rel: Resize the Window relative to its current size.
resize_to: Resize the Window to a specific size.
move_rel: Move the Window relative to its current position.
move_to: Move the Window to a specific position.
h_wnd: Get the handle of the Window.
is_minimized: Check if the Window is minimized.
is_maximized: Check if the Window is maximized.
is_active: Check if the Window is active.
title: Get the title of the Window.
visible: Check if the Window is visible.
left: Get or set the left coordinate of the Window.
right: Get or set the right coordinate of the Window.
top: Get or set the top coordinate of the Window.
bottom: Get or set the bottom coordinate of the Window.
top_left: Get or set the top-left coordinate of the Window.
bottom_left: Get or set the bottom-left coordinate of the Window.
top_right: Get or set the top-right coordinate of the Window.
bottom_right: Get or set the bottom-right coordinate of the Window.
mid_left: Get or set the mid-left coordinate of the Window.
mid_right: Get or set the mid-right coordinate of the Window.
mid_top: Get or set the mid-top coordinate of the Window.
mid_bottom: Get or set the mid-bottom coordinate of the Window.
center: Get or set the center coordinate of the Window.
center_x: Get or set the x-coordinate of the center of the Window.
center_y: Get or set the y-coordinate of the center of the Window.
width: Get or set the width of the Window.
height: Get or set the height of the Window.
size: Get or set the size of the Window.
area: Get the area of the Window.
box: Get or set the box of the Window.


**Methods:**


### .close
```python
.close()
```

---
Closes the window.

### .minimize
```python
.minimize()
```

---
Minimizes the window.

### .maximize
```python
.maximize()
```

---
Maximizes the window.

### .restore
```python
.restore()
```

---
Restores the window to its original size and position.

### .activate
```python
.activate()
```

---
Brings the window to the foreground.

### .resize_rel
```python
.resize_rel(
   width_offset, height_offset
)
```

---
Resizes the window relative to its current size.


**Args**

* **width_offset**  : The offset to add to the window's current width.
* **height_offset**  : The offset to add to the window's current height.


### .resize_to
```python
.resize_to(
   new_width, new_height
)
```

---
Resizes the window to a specific size.


**Args**

* **new_width**  : The new width of the window.
* **new_height**  : The new height of the window.


### .move_rel
```python
.move_rel(
   x_offset, y_offset
)
```

---
Moves the window relative to its current position.


**Args**

* **x_offset**  : The offset to add to the window's current x-coordinate.
* **y_offset**  : The offset to add to the window's current y-coordinate.


### .move_to
```python
.move_to(
   new_left, new_top
)
```

---
Moves the window to a specific position.


**Args**

* **new_left**  : The new left position of the window.
* **new_top**  : The new top position of the window.


### .h_wnd
```python
.h_wnd()
```

---
Gets the handle to the window.


**Returns**

The handle to the window.

### .is_minimized
```python
.is_minimized()
```

---
Checks whether the window is minimized.


**Returns**

True if the window is minimized, False otherwise.

### .is_maximized
```python
.is_maximized()
```

---
Checks whether the window is maximized.


**Returns**

True if the window is maximized, False otherwise.

### .is_active
```python
.is_active()
```

---
Checks whether the window is the active window.


**Returns**

True if the window is the active window, False otherwise.

### .title
```python
.title()
```

---
Gets the title of the window.


**Returns**

The title of the window.

### .visible
```python
.visible()
```

---
Checks whether the window is visible.


**Returns**

True if the window is visible, False otherwise.

### .left
```python
.left()
```

---
Gets the left position of the window.


**Returns**

The left position of the window.

### .right
```python
.right()
```

---
Gets the right position of the window.


**Returns**

The right position of the window.

### .top
```python
.top()
```

---
Gets the top position of the window.


**Returns**

The top position of the window.

### .bottom
```python
.bottom()
```

---
Gets the bottom position of the window.


**Returns**

The bottom position of the window.

### .top_left
```python
.top_left()
```

---
Gets the top-left position of the window.


**Returns**

The top-left position of the window.

### .top_right
```python
.top_right()
```

---
Gets the top-right position of the window.


**Returns**

The top-right position of the window.

### .bottom_left
```python
.bottom_left()
```

---
Gets the bottom-left position of the window.


**Returns**

The bottom-left position of the window.

### .bottom_right
```python
.bottom_right()
```

---
Gets the bottom-right position of the window.


**Returns**

The bottom-right position of the window.

### .mid_left
```python
.mid_left()
```

---
Gets the middle-left position of the window.


**Returns**

The middle-left position of the window.

### .mid_right
```python
.mid_right()
```

---
Gets the middle-right position of the window.


**Returns**

The middle-right position of the window.

### .mid_top
```python
.mid_top()
```

---
Getter for the mid_top property of the Window instance.


**Returns**

* **tuple**  : The mid-top coordinates of the window.


### .mid_bottom
```python
.mid_bottom()
```

---
Getter for the mid_bottom property of the Window instance.


**Returns**

* **tuple**  : The mid-bottom coordinates of the window.


### .center
```python
.center()
```

---
Getter for the center property of the Window instance.


**Returns**

* **tuple**  : The center coordinates of the window.


### .center_x
```python
.center_x()
```

---
Getter for the center_x property of the Window instance.


**Returns**

* **int**  : The x-coordinate of the center of the window.


### .center_y
```python
.center_y()
```

---
Getter for the center_y property of the Window instance.


**Returns**

* **int**  : The y-coordinate of the center of the window.


### .width
```python
.width()
```

---
Getter for the width property of the Window instance.


**Returns**

* **int**  : The width of the window.


### .height
```python
.height()
```

---
Getter for the height property of the Window instance.


**Returns**

* **int**  : The height of the window.


### .size
```python
.size()
```

---
Getter for the size property of the Window instance.


**Returns**

* **tuple**  : The size of the window as a tuple (width, height).


### .area
```python
.area()
```

---
Getter for the area property of the Window instance.


**Returns**

* **int**  : The area of the window.


### .box
```python
.box()
```

---
Getter for the box property of the Window instance.


**Returns**

* **tuple**  : The box of the window as a tuple (left, top, width, height).


----


### _check_for_int_or_float
```python
._check_for_int_or_float(
   arg
)
```

---
Check if the argument is an integer or a float.


**Args**

* **arg**  : The argument to check.


----


### _check_for_two_int_or_float_tuple
```python
._check_for_two_int_or_float_tuple(
   arg
)
```

---
Check if the argument is a tuple of two integers or floats.


**Args**

* **arg**  : The argument to check.


----


### _check_for_four_int_or_float_tuple
```python
._check_for_four_int_or_float_tuple(
   arg
)
```

---
Check if the argument is a tuple of four integers or floats.


**Args**

* **arg**  : The argument to check.


----


### _format_message
```python
._format_message(
   error_code
)
```

---
Format a Windows error message.


**Args**

* **error_code** (int) : The error code to format.


**Returns**

* **str**  : The formatted error message.


----


### _raise_with_last_error
```python
._raise_with_last_error()
```

---
Raise an exception with the last Windows error message.

----


### point_in_rect
```python
.point_in_rect(
   x, y, left, top, width, height
)
```

---
Check if a point is in a rectangle.


**Args**

* **x** (int) : The x-coordinate of the point.
* **y** (int) : The y-coordinate of the point.
* **left** (int) : The left coordinate of the rectangle.
* **top** (int) : The top coordinate of the rectangle.
* **width** (int) : The width of the rectangle.
* **height** (int) : The height of the rectangle.


**Returns**

* **bool**  : True if the point is in the rectangle, False otherwise.


----


### _get_all_titles
```python
._get_all_titles()
```

---
Get the titles of all visible windows.


**Returns**

* **list**  : A list of window titles.


----


### get_windows_at
```python
.get_windows_at(
   x, y
)
```

---
Get all windows at a specific point.


**Args**

* **x** (int) : The x-coordinate of the point.
* **y** (int) : The y-coordinate of the point.


**Returns**

* **list**  : A list of windows at the point.


----


### get_windows_with_title
```python
.get_windows_with_title(
   title
)
```

---
Get all windows with a specific title.


**Args**

* **title** (str) : The title to search for.


**Returns**

* **list**  : A list of windows with the title.


----


### get_all_titles
```python
.get_all_titles()
```

---
Get the titles of all windows.


**Returns**

* **list**  : A list of window titles.


----


### get_all_windows
```python
.get_all_windows()
```

---
Get all windows.


**Returns**

* **list**  : A list of all windows.


----


### get_window_rect
```python
.get_window_rect(
   h_wnd
)
```

---
Get the rectangle of a window.


**Args**

* **h_wnd** (int) : The handle of the window.


**Returns**

* **Rect**  : The rectangle of the window.


----


### get_active_window
```python
.get_active_window()
```

---
Get the active window.


**Returns**

* **Window**  : The active window.


----


### get_window_text
```python
.get_window_text(
   h_wnd
)
```

---
Get the text of a window.


**Args**

* **h_wnd** (int) : The handle of the window.


**Returns**

* **str**  : The text of the window.

