#


## Arrow
```python 
Arrow()
```



----


## ChessEnv
```python 

```




**Methods:**


### .copy
```python
.copy(
   *, stack = True
)
```


### .reset
```python
.reset()
```


### .get_observation
```python
.get_observation(
   orientation = WHITE, flipped = False, mode = 'str'
)
```


### .get_reward
```python
.get_reward(
   piece
)
```


### .step
```python
.step(
   move, move_type = 'uci'
)
```


### .result
```python
.result(
   *, claim_draw = False
)
```


### .unicode
```python
.unicode(
   *, invert_color = False, borders = False, empty_square = 'â\xad˜'
)
```


### .render
```python
.render(
   *, orientation = WHITE, flipped = False, mode = 'str'
)
```


----


### arrow
```python
.arrow(
   screen, color, start, end, thickness
)
```


----


### create_move_labels
```python
.create_move_labels()
```

