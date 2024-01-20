#


## TD3Actor
```python 
TD3Actor(
   in_features, out_features
)
```


---
Twin Delayed Deep Deterministic Policy Gradient Actor Model.

This class is used to create the actor model for the TD3 algorithm.
The actor model is responsible for selecting actions based on the current state of the environment.


**Attributes**

* **hidden1** (Linear) : The first hidden layer.
* **hidden2** (Linear) : The second hidden layer.
* **out** (Linear) : The output layer.


**Args**

* **in_features** (int) : The number of input features.
* **out_features** (int) : The number of output features (actions).



**Methods:**


### .forward
```python
.forward(
   state: Tensor
)
```

---
Forward pass through the actor model.

Accepts a state and returns the predicted action.


**Args**

* **state** (Tensor) : The input tensor representing the current state of the environment.


**Returns**

* **Tensor**  : The output tensor representing the predicted action.


----


## TD3Critic
```python 
TD3Critic(
   in_features, out_features
)
```


---
Twin Delayed Deep Deterministic Policy Gradient Critic Model.

This class is used to create the critic model for the TD3 algorithm.
The critic model is responsible for evaluating the value of the selected action.


**Attributes**

* **hidden1** (Linear) : The first hidden layer.
* **hidden2** (Linear) : The second hidden layer.
* **out** (Linear) : The output layer.


**Args**

* **in_features** (int) : The number of input features.
* **out_features** (int) : The number of output features (actions).



**Methods:**


### .forward
```python
.forward(
   state: Tensor, action: Tensor
)
```

---
Forward pass through the critic model.

Accepts a state and action and returns the value of the selected action.


**Args**

* **state** (Tensor) : The input tensor representing the current state of the environment.
* **action** (Tensor) : The input tensor representing the selected action.


**Returns**

* **Tensor**  : The output tensor representing the value of the selected action.

