#


## DQNModel
```python 
DQNModel(
   lr = 0.001, tau = 0.0001, gamma = 0.99, in_features = 1, out_features = 1
)
```


---
Deep Q Network

# Arguments
models (`keras.nn.Model` instance): See [Model](#) for details.
optimizer (`keras.optimizers.Optimizer` instance):
See [Optimizer](#) for details.
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
.softupdate()
```


### .hardupdate
```python
.hardupdate()
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
   batch = None, update_target = False
)
```


### .evaluate
```python
.evaluate()
```

