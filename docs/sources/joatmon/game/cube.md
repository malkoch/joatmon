#


## Quaternion
```python 
Quaternion(
   x
)
```




**Methods:**


### .as_rotation_matrix
```python
.as_rotation_matrix()
```


### .as_v_theta
```python
.as_v_theta()
```


### .from_v_theta
```python
.from_v_theta(
   cls, v, theta
)
```


### .rotate
```python
.rotate(
   points
)
```


----


## Sticker
```python 
Sticker(
   color
)
```




**Methods:**


### .draw
```python
.draw(
   screen, points, view, rotation, vertical
)
```


----


## Face
```python 
Face(
   n, color, top_left, increment
)
```




**Methods:**


### .draw
```python
.draw(
   screen, view, rotation, vertical
)
```


### .rotate
```python
.rotate(
   times
)
```


### .rotate_layer
```python
.rotate_layer(
   layer
)
```


### .rot90
```python
.rot90()
```


----


## Cube
```python 
Cube(
   n = 3
)
```




**Methods:**


### .draw
```python
.draw()
```


### .swap_faces
```python
.swap_faces(
   faces
)
```


### .swap_layers
```python
.swap_layers(
   faces, layer
)
```


### .u
```python
.u()
```


### .x
```python
.x()
```


### .y
```python
.y()
```


### .z
```python
.z()
```


----


## CubeEnv
```python 

```




**Methods:**


### .close
```python
.close()
```


### .render
```python
.render(
   mode: str = 'human'
)
```


### .reset
```python
.reset()
```


### .seed
```python
.seed(
   seed = None
)
```


### .step
```python
.step(
   action
)
```


----


### flip
```python
.flip(
   pos
)
```


----


### project_points
```python
.project_points(
   points, q, view, vertical = (0, 1, 0)
)
```


----


### resize
```python
.resize(
   pos
)
```

