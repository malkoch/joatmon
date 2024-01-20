#


## Camera
```python 

```


---
Camera class that provides the functionality for capturing video from a camera.


**Attributes**

* **cam** (cv2.VideoCapture) : The camera object.
* **stop_event** (threading.Event) : The event to stop the camera.

---
Methods:
    stop: Stops the camera.


**Methods:**


### .frame
```python
.frame()
```

---
Captures a single frame from the camera.


**Returns**

* **ndarray**  : The captured frame.


### .shot
```python
.shot(
   path
)
```

---
Captures a single frame from the camera and saves it to a file.


**Args**

* **path** (str) : The path to save the frame.


### .stream
```python
.stream()
```

---
Streams video from the camera.


**Yields**

* **ndarray**  : The current frame.


### .record
```python
.record(
   path, time
)
```

---
Records video from the camera for a specified amount of time.


**Args**

* **path** (str) : The path to save the video.
* **time** (int) : The amount of time to record video.


### .stop
```python
.stop()
```

---
Stops the camera.
