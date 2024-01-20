#


## POINT
```python 
POINT()
```


---
A class used to represent a point with x and y coordinates.

Attributes
----------
x : ctypes.c_long
The x-coordinate of the point.
---
    The y-coordinate of the point.

----


### resolution
```python
.resolution()
```

---
Gets the resolution of the system.


**Returns**

* **tuple**  : The resolution of the system as a tuple (height, width).


----


### cursor
```python
.cursor()
```

---
Gets the current position of the cursor.


**Returns**

* **tuple**  : The current position of the cursor as a tuple (y, x).


----


### grab
```python
.grab(
   region = None
)
```

---
Grabs a screenshot of the specified region.


**Args**

* **region** (tuple, optional) : The region to grab a screenshot of as a tuple (left, top, right, bottom). If not specified, a screenshot of the entire screen is grabbed.


**Returns**

* **ndarray**  : The screenshot as a numpy array.

