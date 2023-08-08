#


## CoreNetwork
```python 

```


---
Abstract base class for all implemented nn.

Do not use this abstract base class directly
but instead use one of the concrete nn implemented.

To implement your own nn, you have to implement the following methods:

- `act`
- `replay`
- `load`
- `save`


**Methods:**


### .load
```python
.load()
```

---
load

### .save
```python
.save()
```

---
save

### .predict
```python
.predict()
```

---
Get the action for given state.

Accepts a state and returns an abstract action.

# Arguments
state (abstract): Current state of the game.

---
# Returns
    action (abstract): Network's predicted action for given state.

### .train
```python
.train()
```

---
Train the nn with given batch.

# Arguments
batch (abstract): Mini Batch from Experience Replay Memory.

### .evaluate
```python
.evaluate()
```

---
evaluate
