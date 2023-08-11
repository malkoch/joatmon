#


## CoreRandom
```python 

```


---
Abstract base class for all implemented random processes.

Do not use this abstract base class directly but instead
use one of the concrete random processes implemented.

To implement your own random processes,
you have to implement the following methods:

- `decay`
- `sample`
- `reset`


**Methods:**


### .reset
```python
.reset()
```

---
Reset random state.

### .decay
```python
.decay()
```

---
decay

### .sample
```python
.sample()
```

---
Sample random state.

# Returns
sample (abstract): Random state.

----


## GaussianRandom
```python 
GaussianRandom(
   mu = 0.0, size = 2, sigma = 0.1, sigma_min = 0.01, decay_steps = 200000
)
```


---
Gaussian Noise

# Arguments
mu (float): .
size (int): .
sigma (float): .
sigma_min (float): .
decay_steps (int): .


**Methods:**


### .reset
```python
.reset()
```


### .decay
```python
.decay()
```


### .sample
```python
.sample()
```


----


## OrnsteinUhlenbeck
```python 
OrnsteinUhlenbeck(
   dt = 1.0, mu = 0.0, size = 2, sigma = 0.1, theta = 0.15, sigma_min = 0.01,
   decay_steps = 200000
)
```


---
Ornstein Uhlenbeck Process

# Arguments
dt (float): .
mu (float): .
size (int): .
sigma (float): .
theta (float): .
sigma_min (float): .
decay_steps (int): .


**Methods:**


### .reset
```python
.reset()
```


### .decay
```python
.decay()
```


### .sample
```python
.sample()
```

