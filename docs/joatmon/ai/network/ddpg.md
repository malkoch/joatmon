#


## DDPGActor
```python 
DDPGActor(
   in_features, out_features
)
```


---
Deep Deterministic Policy Gradient Actor Model.

This class is used to create the actor model for the DDPG algorithm.
The actor model is responsible for selecting actions based on the current state of the environment.


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
Forward pass through the actor model.


**Args**

* **x** (Tensor) : The input tensor representing the current state of the environment.


**Returns**

* **Tensor**  : The output tensor representing the selected action.


----


## DDPGCritic
```python 
DDPGCritic(
   in_features, out_features
)
```


---
Deep Deterministic Policy Gradient Critic Model.

This class is used to create the critic model for the DDPG algorithm.
The critic model is responsible for evaluating the value of the selected action.


**Attributes**

* **extractor** (Sequential) : A sequence of convolutional layers used for feature extraction.
* **relu** (ReLU) : The ReLU activation function.
* **linear1** (Linear) : The first linear layer.
* **linear2** (Linear) : The second linear layer.
* **linear3** (Linear) : The third linear layer.
* **bn1** (BatchNorm) : The first batch normalization layer.
* **bn2** (BatchNorm) : The second batch normalization layer.


**Args**

* **in_features** (int) : The number of input features.
* **out_features** (int) : The number of output features (actions).



**Methods:**


### .forward
```python
.forward(
   x, y
)
```

---
Forward pass through the critic model.


**Args**

* **x** (Tensor) : The input tensor representing the current state of the environment.
* **y** (Tensor) : The input tensor representing the selected action.


**Returns**

* **Tensor**  : The output tensor representing the value of the selected action.

