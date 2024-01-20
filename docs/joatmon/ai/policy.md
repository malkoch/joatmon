#


## CorePolicy
```python 

```


---
Abstract base class for all implemented policies.

This class should not be used directly. Instead, use one of the concrete policies implemented.
To implement your own policy, you have to implement the following methods: `decay`, `reset`, `use_network`.


**Methods:**


### .reset
```python
.reset()
```

---
Reset the policy.

This method should be overridden by any subclass.

### .decay
```python
.decay()
```

---
Decay the policy.

This method should be overridden by any subclass.

### .use_network
```python
.use_network()
```

---
Determine whether to use the network for decision making.

This method should be overridden by any subclass.


**Returns**

* **use** (bool) : Boolean value for using the network.


----


## GreedyQPolicy
```python 

```


---
Greedy Q Policy

This class implements a policy that always selects the action with the highest expected reward.


**Methods:**


### .reset
```python
.reset()
```

---
Reset the policy.

This method is currently a placeholder and does nothing.

### .decay
```python
.decay()
```

---
Decay the policy.

This method is currently a placeholder and does nothing.

### .use_network
```python
.use_network()
```

---
Determine whether to use the network for decision making.

For a GreedyQPolicy, this always returns True.


**Returns**

* **use** (bool) : Boolean value for using the network.


----


## EpsilonGreedyPolicy
```python 
EpsilonGreedyPolicy(
   max_value = 1.0, min_value = 0.0, decay_steps = 1
)
```


---
Epsilon Greedy Policy

This class implements a policy that selects a random action with probability epsilon and the action with the highest expected reward with probability 1 - epsilon.


**Attributes**

* **epsilon** (float) : The probability of selecting a random action.
* **min_value** (float) : The minimum value that epsilon can decay to.
* **epsilon_decay** (float) : The amount by which epsilon is reduced at each step.


**Args**

* **max_value** (float) : The initial value of epsilon.
* **min_value** (float) : The minimum value that epsilon can decay to.
* **decay_steps** (int) : The number of steps over which epsilon decays from max_value to min_value.



**Methods:**


### .reset
```python
.reset()
```

---
Reset the policy.

This method is currently a placeholder and does nothing.

### .decay
```python
.decay()
```

---
Decay the policy.

This method reduces the value of epsilon by epsilon_decay, down to a minimum of min_value.

### .use_network
```python
.use_network()
```

---
Determine whether to use the network for decision making.

For an EpsilonGreedyPolicy, this returns True with probability 1 - epsilon and False with probability epsilon.


**Returns**

* **use** (bool) : Boolean value for using the network.

