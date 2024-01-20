#


## Mouse
```python 
Mouse()
```


---
A class used to represent a Mouse.

...

Attributes
----------
MOUSE_DOWN : int
The code for the mouse down event.
---
    The code for the mouse up event.
    The code for the mouse left event.
    The code for the mouse right event.
    The code for the mouse middle event.
    The minimum x-coordinate for the mouse.
    The maximum x-coordinate for the mouse.
    The minimum y-coordinate for the mouse.
    The maximum y-coordinate for the mouse.

Methods
-------
restrict(min_x, max_x, min_y, max_y)
    Restricts the mouse movement to a specific area.
move_to(x=None, y=None)
    Moves the mouse to the specified coordinates.
mouse_down(x=None, y=None, button=None)
    Simulates a mouse down event at the specified coordinates.
mouse_up(x=None, y=None, button=None)
    Simulates a mouse up event at the specified coordinates.
click(x=None, y=None, button=None)
    Simulates a mouse click event at the specified coordinates.


**Methods:**


### .restrict
```python
.restrict(
   min_x, max_x, min_y, max_y
)
```

---
Restricts the mouse movement to a specific area.


**Args**

* **min_x** (int) : The minimum x-coordinate for the mouse.
* **max_x** (int) : The maximum x-coordinate for the mouse.
* **min_y** (int) : The minimum y-coordinate for the mouse.
* **max_y** (int) : The maximum y-coordinate for the mouse.


### .mouse_down
```python
.mouse_down(
   x = None, y = None, button = None
)
```

---
Simulates a mouse down event at the specified coordinates.


**Args**

* **x** (int, optional) : The x-coordinate for the mouse down event. If not specified, the mouse's current x-coordinate is used.
* **y** (int, optional) : The y-coordinate for the mouse down event. If not specified, the mouse's current y-coordinate is used.
* **button** (int, optional) : The button for the mouse down event. If not specified, the left mouse button is used.


### .mouse_up
```python
.mouse_up(
   x = None, y = None, button = None
)
```

---
Simulates a mouse up event at the specified coordinates.


**Args**

* **x** (int, optional) : The x-coordinate for the mouse up event. If not specified, the mouse's current x-coordinate is used.
* **y** (int, optional) : The y-coordinate for the mouse up event. If not specified, the mouse's current y-coordinate is used.
* **button** (int, optional) : The button for the mouse up event. If not specified, the left mouse button is used.


### .click
```python
.click(
   x = None, y = None, button = None
)
```

---
Simulates a mouse click event at the specified coordinates.


**Args**

* **x** (int, optional) : The x-coordinate for the mouse click event. If not specified, the mouse's current x-coordinate is used.
* **y** (int, optional) : The y-coordinate for the mouse click event. If not specified, the mouse's current y-coordinate is used.
* **button** (int, optional) : The button for the mouse click event. If not specified, the left mouse button is used.


----


### _send_mouse_event
```python
._send_mouse_event(
   x, y, event
)
```

---
Sends a mouse event to the specified coordinates.


**Args**

* **x** (int) : The x-coordinate for the mouse event.
* **y** (int) : The y-coordinate for the mouse event.
* **event** (int) : The type of mouse event.

