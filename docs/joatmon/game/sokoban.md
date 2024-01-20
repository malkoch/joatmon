#


## SokobanEnv
```python 
SokobanEnv(
   xml, xmls, sprites
)
```


---
The SokobanEnv class is a subclass of the CoreEnv class. It represents the environment for the Sokoban game.
This class is responsible for initializing the game environment, handling game logic, and rendering the game state.


**Attributes**

* **xml** (str) : The name of the XML file containing the environment specifications.
* **xmls** (str) : The path to the directory containing the XML files.
* **sprites** (str) : The path to the directory containing the sprite images.



**Methods:**


### ._get_distance
```python
._get_distance(
   shape, shapes, method = 'e'
)
```

---
This method calculates the minimum distance between a shape and a list of shapes.
The distance can be calculated using either the Euclidean distance or the Manhattan distance.

### ._post_step_callback
```python
._post_step_callback(
   space, key
)
```

---
This method is called after the physics engine has processed a step.
It sets the angle of the body to 0 and reindexes the shapes for the body in the space.


**Args**

* **space** (pymunk.Space) : The physics space.
* **key** (pymunk.Shape) : The shape that was processed in the step.


### .close
```python
.close()
```

---
This method closes the game window and quits pygame.

### .render
```python
.render(
   mode = 'human'
)
```

---
This method renders the current game state on the screen.


**Args**

* **mode** (str, optional) : The mode in which to render the game. Default is 'human'.


### .reset
```python
.reset()
```

---
This method resets the game state.
It clears the current game objects, generates a new layout, recreates the game objects, and plays a step.


**Returns**

* **ndarray**  : The observation of the new game state.


### .seed
```python
.seed(
   seed = None
)
```

---
This method sets the seed for the random number generator.


**Args**

* **seed** (int, optional) : The seed to set. If not specified, the random number generator is not seeded.


### .step
```python
.step(
   action
)
```

---
This method simulates a step in the game based on the given action.
It applies a force to the player in the direction of the action, steps the physics space, and returns the new game state.


**Args**

* **action** (tuple) : The action to take, represented as a vector.


**Returns**

* **tuple**  : The observation of the new game state, the reward for the step, whether the game is over, and additional info.


### .goal
```python
.goal()
```

---
This method generates an image of the goal state of the game.
The goal state is where all blocks are on the goals.


**Returns**

* **ndarray**  : The image of the goal state.


----


### generate_room
```python
.generate_room(
   dim = (7, 7), wall_prob = 0.3, p_change_directions = 0.35, num_steps = 5,
   num_boxes = 1, tries = 4
)
```

---
Generates a room for the Sokoban game.


**Args**

* **dim** (tuple) : The dimensions of the room. Default is (7, 7).
* **wall_prob** (float) : The probability of a wall being placed in a cell. Default is 0.3.
* **p_change_directions** (float) : The probability of changing directions while generating the room. Default is 0.35.
* **num_steps** (int) : The number of steps to take while generating the room. Default is 5.
* **num_boxes** (int) : The number of boxes to place in the room. Default is 1.
* **tries** (int) : The number of attempts to generate a valid room. Default is 4.


**Returns**

* **tuple**  : The room structure, room state, and box mapping.


----


### create_circle_body
```python
.create_circle_body(
   mass, body_type, radius
)
```

---
Creates a circular body for the physics engine.


**Args**

* **mass** (float) : The mass of the body.
* **body_type** (pymunk.Body.body_type) : The type of the body (static, dynamic, or kinematic).
* **radius** (float) : The radius of the body.


**Returns**

* **Body**  : The created body.


----


### create_circle_shape
```python
.create_circle_shape(
   body, radius, friction, elasticity, collision_type, sensor
)
```

---
Creates a circular shape for the physics engine.


**Args**

* **body** (pymunk.Body) : The body to which the shape is attached.
* **radius** (float) : The radius of the shape.
* **friction** (float) : The friction coefficient of the shape.
* **elasticity** (float) : The elasticity of the shape.
* **collision_type** (int) : The collision type of the shape.
* **sensor** (bool) : Whether the shape is a sensor.


**Returns**

* **Circle**  : The created shape.


----


### create_rectangle_body
```python
.create_rectangle_body(
   mass, body_type, half_size
)
```

---
Creates a rectangular body for the physics engine.


**Args**

* **mass** (float) : The mass of the body.
* **body_type** (pymunk.Body.body_type) : The type of the body (static, dynamic, or kinematic).
* **half_size** (float) : Half the size of the body.


**Returns**

* **tuple**  : The created body and the points defining the rectangle.


----


### create_rectangle_shape
```python
.create_rectangle_shape(
   body, points, friction, elasticity, collision_type, sensor
)
```

---
Creates a rectangular shape for the physics engine.


**Args**

* **body** (pymunk.Body) : The body to which the shape is attached.
* **points** (list) : The points defining the rectangle.
* **friction** (float) : The friction coefficient of the shape.
* **elasticity** (float) : The elasticity of the shape.
* **collision_type** (int) : The collision type of the shape.
* **sensor** (bool) : Whether the shape is a sensor.


**Returns**

* **Poly**  : The created shape.


----


### draw_circle
```python
.draw_circle(
   screen, color, position, radius
)
```

---
Draws a circle on the screen.


**Args**

* **screen** (pygame.Surface) : The surface on which to draw the circle.
* **color** (tuple) : The color of the circle.
* **position** (tuple) : The position of the center of the circle.
* **radius** (int) : The radius of the circle.


----


### draw_rectangle
```python
.draw_rectangle(
   screen, color, position, half_size
)
```

---
Draws a rectangle on the screen.


**Args**

* **screen** (pygame.Surface) : The surface on which to draw the rectangle.
* **color** (tuple) : The color of the rectangle.
* **position** (tuple) : The position of the center of the rectangle.
* **half_size** (int) : Half the size of the rectangle.


----


### draw_sprite
```python
.draw_sprite(
   screen, image, position, half_size
)
```

---
Draws a sprite on the screen.


**Args**

* **screen** (pygame.Surface) : The surface on which to draw the sprite.
* **image** (pygame.Surface) : The image to draw.
* **position** (tuple) : The position at which to draw the sprite.
* **half_size** (int) : Half the size of the sprite.


----


### flip_y
```python
.flip_y(
   vector, y
)
```

---
Flips the y-coordinate of a vector.


**Args**

* **vector** (Vec2d) : The vector to flip.
* **y** (int) : The height of the screen.


**Returns**

* **Vec2d**  : The vector with the y-coordinate flipped.


----


### layout_getter
```python
.layout_getter(
   layout_specs
)
```

---
Returns a function that generates a room layout based on the given specifications.


**Args**

* **layout_specs** (dict) : The specifications for the room layout.


**Returns**

* **function**  : A function that generates a room layout.


----


### load_sprite
```python
.load_sprite(
   path, color, size = None
)
```

---
Loads a sprite from a file, or creates a new sprite if the file does not exist.


**Args**

* **path** (str) : The path to the sprite file.
* **color** (tuple) : The color to use if the sprite file does not exist.
* **size** (tuple or int, optional) : The size of the sprite. If not specified, the original size of the sprite is used.


**Returns**

* **Surface**  : The loaded or created sprite.


----


### load_xml
```python
.load_xml(
   element
)
```

---
Loads an XML file or parses an XML string.


**Args**

* **element** (str or xml.etree.ElementTree.Element) : The XML file path or XML string.


**Returns**

* **dict**  : A dictionary representation of the XML.


----


### euclidean_distance
```python
.euclidean_distance(
   position1, position2
)
```

---
Calculates the Euclidean distance between two positions.


**Args**

* **position1** (Vec2d) : The first position, represented as a vector.
* **position2** (Vec2d) : The second position, represented as a vector.


**Returns**

* **float**  : The Euclidean distance between the two positions.


----


### manhattan_distance
```python
.manhattan_distance(
   position1, position2
)
```

---
Calculates the Manhattan distance between two positions.


**Args**

* **position1** (Vec2d) : The first position, represented as a vector.
* **position2** (Vec2d) : The second position, represented as a vector.


**Returns**

* **float**  : The Manhattan distance between the two positions.


----


### convert_to
```python
.convert_to(
   value, type_
)
```

---
Converts a value to a specified type.


**Args**

* **value** (any) : The value to convert.
* **type_** (type) : The type to convert the value to.


**Returns**

* **any**  : The converted value.


**Raises**

* **ValueError**  : If the value cannot be converted to the specified type.

