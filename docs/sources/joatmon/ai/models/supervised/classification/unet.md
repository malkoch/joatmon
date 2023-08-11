#


## UNetModel
```python 
UNetModel(
   lr = 0.001, channels = 3, classes = 10
)
```


---
U Network

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

