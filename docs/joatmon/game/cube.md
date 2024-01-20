#


## Quaternion
```python 
Quaternion(
   x
)
```


---
Quaternion class for representing and manipulating quaternions.

Quaternions are a number system that extends the complex numbers. They are used for calculations involving three-dimensional rotations.


**Attributes**

* **x** (ndarray) : The quaternion's components.



**Methods:**


### .as_rotation_matrix
```python
.as_rotation_matrix()
```

---
Convert the quaternion to a rotation matrix.


**Returns**

* **ndarray**  : The rotation matrix.


### .as_v_theta
```python
.as_v_theta()
```

---
Convert the quaternion to a vector and an angle.


**Returns**

* **tuple**  : The vector and the angle.


### .from_v_theta
```python
.from_v_theta(
   cls, v, theta
)
```

---
Create a quaternion from a vector and an angle.


**Args**

* **v** (array_like) : The vector.
* **theta** (array_like) : The angle.


**Returns**

* **Quaternion**  : The created quaternion.


### .rotate
```python
.rotate(
   points
)
```

---
Rotate points in 3D space using the quaternion.


**Args**

* **points** (array_like) : The points in 3D space.


**Returns**

* **ndarray**  : The rotated points.


----


## Sticker
```python 
Sticker(
   color
)
```


---
Sticker class for representing and manipulating stickers on a Rubik's cube.


**Attributes**

* **color** (int) : The color of the sticker.



**Methods:**


### .draw
```python
.draw(
   screen, points, view, rotation, vertical
)
```

---
Draw the sticker on the screen.


**Args**

* **screen** (Surface) : The Pygame surface to draw on.
* **points** (array_like) : The points defining the sticker's shape.
* **view** (array_like) : The view vector.
* **rotation** (Quaternion) : The rotation quaternion.
* **vertical** (array_like) : The vertical vector.


----


## Face
```python 
Face(
   n, color, top_left, increment
)
```


---
Face class for representing and manipulating faces of a Rubik's cube.


**Attributes**

* **n** (int) : The size of the face.
* **top_left** (ndarray) : The top left corner of the face.
* **increment** (tuple) : The increment for each sticker on the face.
* **stickers** (list) : The stickers on the face.



**Methods:**


### .draw
```python
.draw(
   screen, view, rotation, vertical
)
```

---
Draw the face on the screen.


**Args**

* **screen** (Surface) : The Pygame surface to draw on.
* **view** (array_like) : The view vector.
* **rotation** (Quaternion) : The rotation quaternion.
* **vertical** (array_like) : The vertical vector.


### .rotate
```python
.rotate(
   times
)
```

---
Rotate the face a certain number of times.


**Args**

* **times** (int) : The number of times to rotate the face.


### .rotate_layer
```python
.rotate_layer(
   layer
)
```

---
Rotate a layer of the face.


**Args**

* **layer** (int) : The layer of the face to rotate.


### .rot90
```python
.rot90()
```

---
Rotate the face 90 degrees.

----


## Cube
```python 
Cube(
   n = 3
)
```


---
Cube class for representing and manipulating a Rubik's cube.


**Attributes**

* **n** (int) : The size of the cube.
* **faces** (dict) : The faces of the cube.
* **order** (list) : The order of the faces.
* **front** (Surface) : The Pygame surface for the front of the cube.
* **back** (Surface) : The Pygame surface for the back of the cube.
* **screen** (Surface) : The Pygame surface for the screen.
* **view** (tuple) : The view vector.
* **rotation** (Quaternion) : The rotation quaternion.
* **vertical** (list) : The vertical vector.



**Methods:**


### .draw
```python
.draw()
```

---
Draw the cube on the screen.

### .swap_faces
```python
.swap_faces(
   faces
)
```

---
Swap the colors of the stickers on the specified faces.


**Args**

* **faces** (list) : The faces to swap.


### .swap_layers
```python
.swap_layers(
   faces, layer
)
```

---
Swap the colors of the stickers on the specified layers of the faces.


**Args**

* **faces** (list) : The faces whose layers to swap.
* **layer** (int) : The layer to swap.


### .u
```python
.u()
```

---
Rotate the 'U' face of the cube and adjust the colors of the stickers accordingly.

### .x
```python
.x()
```

---
Rotate the cube around the x-axis and adjust the colors of the stickers accordingly.

### .y
```python
.y()
```

---
Rotate the cube around the y-axis and adjust the colors of the stickers accordingly.

### .z
```python
.z()
```

---
Rotate the cube around the z-axis and adjust the colors of the stickers accordingly.

----


## CubeEnv
```python 

```


---
CubeEnv class for creating a Rubik's cube environment.

This class inherits from CoreEnv and provides a skeleton for the methods that need to be implemented in the subclasses.


**Methods:**


### .close
```python
.close()
```

---
Clean up the environment's resources.


**Raises**

* **NotImplementedError**  : This method needs to be implemented in the subclasses.


### .render
```python
.render(
   mode: str = 'human'
)
```

---
Render the environment.


**Args**

* **mode** (str, optional) : The mode to use for rendering. Defaults to 'human'.


**Raises**

* **NotImplementedError**  : This method needs to be implemented in the subclasses.


### .reset
```python
.reset()
```

---
Reset the environment to its initial state and return the initial observation.


**Raises**

* **NotImplementedError**  : This method needs to be implemented in the subclasses.


### .seed
```python
.seed(
   seed = None
)
```

---
Set the seed for the environment's random number generator.


**Args**

* **seed** (int, optional) : The seed to use. Defaults to None.


**Raises**

* **NotImplementedError**  : This method needs to be implemented in the subclasses.


### .step
```python
.step(
   action
)
```

---
Run one timestep of the environment's dynamics.


**Args**

* **action**  : An action to take in the environment.


**Raises**

* **NotImplementedError**  : This method needs to be implemented in the subclasses.


----


### flip
```python
.flip(
   pos
)
```

---
Flip the position coordinates.

This function flips the y-coordinate of the position and adds the screen size to it.


**Args**

* **pos** (tuple) : The position coordinates.


**Returns**

* **tuple**  : The flipped position coordinates.


----


### project_points
```python
.project_points(
   points, q, view, vertical = (0, 1, 0)
)
```

---
Project points in 3D space to 2D space.

This function projects points in 3D space to 2D space using a quaternion for rotation, a view vector, and a vertical vector.


**Args**

* **points** (array_like) : The points in 3D space.
* **q** (Quaternion) : The quaternion for rotation.
* **view** (array_like) : The view vector.
* **vertical** (array_like, optional) : The vertical vector. Defaults to (0, 1, 0).


**Returns**

* **ndarray**  : The projected points in 2D space.


----


### resize
```python
.resize(
   pos
)
```

---
Resize the position coordinates.

This function resizes the position coordinates by multiplying them with the screen size divided by 4 and adding the screen size divided by 2.


**Args**

* **pos** (tuple) : The position coordinates.


**Returns**

* **tuple**  : The resized position coordinates.

