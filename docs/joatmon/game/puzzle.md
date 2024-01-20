#


## Puzzle2048
```python 
Puzzle2048(
   size
)
```


---
Puzzle2048 class for creating a 2048 game environment.

This class inherits from CoreEnv and provides methods for game operations such as resetting the game, rendering the game state, and performing a step in the game.


**Attributes**

* **size** (int) : The size of the game grid.
* **matrix** (list) : The game grid represented as a 2D list.



**Methods:**


### .close
```python
.close()
```

---
Closes the game environment.

### .render
```python
.render(
   mode = 'human'
)
```

---
Renders the game state.


**Args**

* **mode** (str, optional) : The mode to use for rendering. Defaults to 'human'.


### .reset
```python
.reset()
```

---
Resets the game state.


**Returns**

* **list**  : The reset game grid.


### .seed
```python
.seed(
   seed = None
)
```

---
Sets the seed for the game's random number generator.


**Args**

* **seed** (int, optional) : The seed to use. Defaults to None.


### .step
```python
.step(
   action
)
```

---
Performs a step in the game.


**Args**

* **action** (int) : The action to perform.


**Returns**

* **tuple**  : The new game state, the reward obtained, whether the game is over, and additional info.

