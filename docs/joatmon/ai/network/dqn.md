#


## DQN
```python 
DQN(
   in_features, out_features
)
```


---
Deep Q-Network Model.

This class is used to create the DQN model for the DQN algorithm.
The DQN model is responsible for selecting actions based on the current state of the environment.


**Attributes**

* **extractor** (Sequential) : A sequence of convolutional layers used for feature extraction.
* **predictor** (Sequential) : A sequence of linear layers used for action prediction.


**Args**

* **in_features** (int) : The number of input features.
* **out_features** (int) : The number of output features (actions).



**Methods:**


### .forward
```python
.forward(
   x
)
```

---
Forward pass through the DQN model.

Accepts a state and returns the predicted action.


**Args**

* **x** (Tensor) : The input tensor representing the current state of the environment.


**Returns**

* **Tensor**  : The output tensor representing the predicted action.

