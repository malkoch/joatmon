#


## DQNModel
```python 
DQNModel(
   lr = 0.001, tau = 0.0001, gamma = 0.99, in_features = 1, out_features = 1
)
```


---
Deep Q Network Model.

This class implements the DQN model, which is a type of reinforcement learning model.
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
Load the local and target networks from the specified path.

# Arguments
path (str): The path to the directory where the networks are stored.

### .save
```python
.save(
   path = ''
)
```

---
Save the local and target networks to the specified path.

# Arguments
path (str): The path to the directory where the networks should be saved.

### .initialize
```python
.initialize(
   w_init = None, b_init = None
)
```

---
Initialize the weights and biases of the local and target networks.

# Arguments
w_init (callable, optional): The function to use for initializing the weights.
b_init (callable, optional): The function to use for initializing the biases.

### .softupdate
```python
.softupdate()
```

---
Perform a soft update of the target network parameters.

### .hardupdate
```python
.hardupdate()
```

---
Perform a hard update of the target network parameters.

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
   batch = None, update_target = False
)
```

---
Train the local network with a given batch.

# Arguments
batch (tuple of array-like, optional): The batch of experiences to train on.
update_target (bool, optional): Whether to perform a soft update of the target network.

---
# Returns
    loss (float): The loss of the local network.

### .evaluate
```python
.evaluate()
```

---
Evaluate the local and target networks.

This method should be overridden in a subclass.
