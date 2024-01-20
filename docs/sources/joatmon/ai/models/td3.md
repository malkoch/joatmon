#


## TD3Model
```python 
TD3Model(
   lr = 0.001, tau = 0.0001, gamma = 0.99, in_features = 1, out_features = 1
)
```


---
Twin Delayed Deep Deterministic Policy Gradient Model.

This class implements the TD3 model, which is a type of reinforcement learning model.
It inherits from the CoreModel class and overrides its abstract methods.

# Arguments
lr (float): The learning rate for the Adam optimizer.
tau (float): The factor for soft update of target parameters.
gamma (float): The discount factor.
in_features (int): The number of input features.
out_features (int): The number of output features.


**Methods:**


### .load
```python
.load(
   path = ''
)
```

---
Load the actor and critic networks from the specified path.

# Arguments
path (str): The path to the directory where the networks are stored.

### .save
```python
.save(
   path = ''
)
```

---
Save the actor and critic networks to the specified path.

# Arguments
path (str): The path to the directory where the networks should be saved.

### .initialize
```python
.initialize(
   w_init = None, b_init = None
)
```

---
Initialize the weights and biases of the actor and critic networks.

# Arguments
w_init (callable, optional): The function to use for initializing the weights.
b_init (callable, optional): The function to use for initializing the biases.

### .softupdate
```python
.softupdate(
   network: str
)
```

---
Perform a soft update of the target network parameters.

# Arguments
network (str): The name of the network to update ('actor' or 'critic_1' or 'critic_2').

### .hardupdate
```python
.hardupdate(
   network: str
)
```

---
Perform a hard update of the target network parameters.

# Arguments
network (str): The name of the network to update ('actor' or 'critic_1' or 'critic_2').

### .predict
```python
.predict(
   state = None
)
```

---
Get the action for a given state.

# Arguments
state (array-like, optional): The current state.

---
# Returns
    action (array-like): The predicted action for the given state.

### .train
```python
.train(
   batch = None, update_target = True
)
```

---
Train the actor and critic networks with a given batch.

# Arguments
batch (tuple of array-like, optional): The batch of experiences to train on.
update_target (bool, optional): Whether to perform a soft update of the target networks.

---
# Returns
    losses (list of float): The actor and critic losses.

### .evaluate
```python
.evaluate()
```

---
Evaluate the actor and critic networks.

This method should be overridden in a subclass.
