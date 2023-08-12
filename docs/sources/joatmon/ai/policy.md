#


## CorePolicy
```python

```


---
Abstract base class for all implemented policy.

Do not use this abstract base class directly but
instead use one of the concrete policy implemented.

To implement your own policy, you have to implement the following methods:

- `decay`
- `use_network`


**Methods:**


### .reset
```python
.reset()
```

---
reset

### .decay
```python
.decay()
```

---
Decaying the epsilon / sigma value of the policy.

### .use_network
```python
.use_network()
```

---
Sample an experience replay batch with size.

# Returns
use (bool): Boolean value for using the nn.

----


## GreedyQPolicy
```python

```




**Methods:**


### .reset
```python
.reset()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .decay
```python
.decay()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .use_network
```python
.use_network()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


## EpsilonGreedyPolicy
```python
EpsilonGreedyPolicy(
   max_value = 1.0, min_value = 0.0, decay_steps = 1
)
```


---
Epsilon Greedy

# Arguments
max_value (float): .
min_value (float): .
decay_steps (int): .


**Methods:**


### .reset
```python
.reset()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .decay
```python
.decay()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .use_network
```python
.use_network()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.
