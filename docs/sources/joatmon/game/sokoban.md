#


## SokobanEnv
```python 
SokobanEnv(
   xml, xmls, sprites
)
```




**Methods:**


### ._get_distance
```python
._get_distance(
   shape, shapes, method = 'e'
)
```


### ._post_step_callback
```python
._post_step_callback(
   space, key
)
```


### .close
```python
.close()
```


### .render
```python
.render(
   mode = 'human'
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


### .goal
```python
.goal()
```


----


### generate_room
```python
.generate_room(
   dim = (7, 7), wall_prob = 0.3, p_change_directions = 0.35, num_steps = 5,
   num_boxes = 1, tries = 4
)
```


----


### create_circle_body
```python
.create_circle_body(
   mass, body_type, radius
)
```


----


### create_circle_shape
```python
.create_circle_shape(
   body, radius, friction, elasticity, collision_type, sensor
)
```


----


### create_rectangle_body
```python
.create_rectangle_body(
   mass, body_type, half_size
)
```


----


### create_rectangle_shape
```python
.create_rectangle_shape(
   body, points, friction, elasticity, collision_type, sensor
)
```


----


### draw_circle
```python
.draw_circle(
   screen, color, position, radius
)
```


----


### draw_rectangle
```python
.draw_rectangle(
   screen, color, position, half_size
)
```


----


### draw_sprite
```python
.draw_sprite(
   screen, image, position, half_size
)
```


----


### flip_y
```python
.flip_y(
   vector, y
)
```


----


### layout_getter
```python
.layout_getter(
   layout_specs
)
```


----


### load_sprite
```python
.load_sprite(
   path, color, size = None
)
```


----


### load_xml
```python
.load_xml(
   element
)
```


----


### euclidean_distance
```python
.euclidean_distance(
   position1, position2
)
```


----


### manhattan_distance
```python
.manhattan_distance(
   position1, position2
)
```


----


### convert_to
```python
.convert_to(
   value, type_
)
```

