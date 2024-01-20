#


## CoreRandom
```python 

```


---
Process a state.

This method accepts a state and processes it for use in reinforcement learning. The state is resized to 84x84 and the color channels are moved to the second dimension.


**Args**

* **state** (numpy array) : The input state as a numpy array.


**Returns**

* **array**  : The processed state as a numpy array.



**Methods:**


### .reset
```python
.reset()
```

---
Reset the random state.

This method should be overridden by any subclass.

### .decay
```python
.decay()
```

---
Decay the random state.

This method should be overridden by any subclass.

### .sample
```python
.sample()
```

---
Sample from the random state.

This method should be overridden by any subclass.


**Returns**

* **sample** (abstract) : The sampled random state.


----


## GaussianRandom
```python 
GaussianRandom(
   mu = 0.0, size = 2, sigma = 0.1, sigma_min = 0.01, decay_steps = 200000
)
```


---
Gaussian Noise

This class generates a Gaussian noise process.


**Attributes**

* **mu** (float) : The mean of the Gaussian distribution.
* **size** (int) : The size of the output sample.
* **sigma** (float) : The standard deviation of the Gaussian distribution.
* **sigma_min** (float) : The minimum standard deviation.
* **decay_steps** (int) : The number of steps over which the standard deviation decays from sigma to sigma_min.



**Methods:**


### .reset
```python
.reset()
```

---
Reset the random state.

This method resets the number of steps and the current standard deviation.

### .decay
```python
.decay()
```

---
Decay the random state.

This method increments the number of steps and updates the current standard deviation.

### .sample
```python
.sample()
```

---
Sample from the random state.

This method generates a sample from a Gaussian distribution with the current mean and standard deviation.


**Returns**

* **x** (numpy array) : The sampled random state.


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

This class generates an Ornstein Uhlenbeck noise process.


**Attributes**

* **dt** (float) : The time increment.
* **mu** (float) : The mean of the distribution.
* **size** (int) : The size of the output sample.
* **sigma** (float) : The standard deviation of the distribution.
* **theta** (float) : The rate of mean reversion.
* **sigma_min** (float) : The minimum standard deviation.
* **decay_steps** (int) : The number of steps over which the standard deviation decays from sigma to sigma_min.



**Methods:**


### .reset
```python
.reset()
```

---
Reset the random state.

This method resets the number of steps, the current standard deviation, and the previous state.

### .decay
```python
.decay()
```

---
Decay the random state.

This method increments the number of steps and updates the current standard deviation.

### .sample
```python
.sample()
```

---
Sample from the random state.

This method generates a sample from an Ornstein Uhlenbeck process with the current parameters.


**Returns**

* **x** (numpy array) : The sampled random state.

