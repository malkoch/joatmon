#


## DQNModel
```python
DQNModel(
   lr = 0.001, tau = 0.0001, gamma = 0.99, in_features = 1, out_features = 1
)
```


---
Deep Q Network

# Arguments
models (`keras.nn.Model` instance): See [Model](#) for details.
optimizer (`keras.optimizers.Optimizer` instance):
See [Optimizer](#) for details.
tau (float): tau.
gamma (float): gamma.


**Methods:**


### .load
```python
.load(
   path = ''
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .save
```python
.save(
   path = ''
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .initialize
```python
.initialize(
   w_init = None, b_init = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .softupdate
```python
.softupdate()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .hardupdate
```python
.hardupdate()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .predict
```python
.predict(
   state = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .train
```python
.train(
   batch = None, update_target = False
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .evaluate
```python
.evaluate()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.
