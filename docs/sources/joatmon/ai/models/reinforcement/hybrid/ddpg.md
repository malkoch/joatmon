#


## DDPGModel
```python 
DDPGModel(
   lr = 0.001, tau = 0.0001, gamma = 0.99, in_features = 1, out_features = 1
)
```


---
Deep Deterministic Policy Gradient

# Arguments
actor_model (`keras.nn.Model` instance): See [Model](#) for details.
critic_model (`keras.nn.Model` instance): See [Model](#) for details.
optimizer (`keras.optimizers.Optimizer` instance):
See [Optimizer](#) for details.
action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
See [Input](#) for details.
tau (float): tau.
gamma (float): gamma.


**Methods:**


### .load
```python
.load(
   path = ''
)
```


### .save
```python
.save(
   path = ''
)
```


### .initialize
```python
.initialize(
   w_init = None, b_init = None
)
```


### .softupdate
```python
.softupdate(
   network: str
)
```


### .hardupdate
```python
.hardupdate(
   network: str
)
```


### .predict
```python
.predict(
   state = None
)
```


### .train
```python
.train(
   batch = None, update_target = True
)
```


### .evaluate
```python
.evaluate()
```

