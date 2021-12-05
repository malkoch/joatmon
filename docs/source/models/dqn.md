### Introduction

---

<span style="float:right;">[[source]](https://github.com/malkoch/joatmon/blob/master/joatmon/ai/models/dqn.py#L21)</span>
### DQNModel

```python
joatmon.ai.models.dqn.DQNModel.joatmon.ai.models.dqn.DQNModel(lr=0.001, tau=0.0001, gamma=0.99, network=None)
```


Deep Q Network

__Arguments__

- __models__ (`keras.nn.Model` instance): See [Model](#) for details.
- __optimizer__ (`keras.optimizers.Optimizer` instance):
See [Optimizer](#) for details.
- __tau__ (float): tau.
- __gamma__ (float): gamma.

----

<span style="float:right;">[[source]](https://github.com/malkoch/joatmon/blob/master/joatmon/ai/models/dqn.py#L47)</span>

### load


```python
DQNModel.load(self, path='')
```

----

<span style="float:right;">[[source]](https://github.com/malkoch/joatmon/blob/master/joatmon/ai/models/dqn.py#L51)</span>

### save


```python
DQNModel.save(self, path='')
```

----

<span style="float:right;">[[source]](https://github.com/malkoch/joatmon/blob/master/joatmon/ai/models/dqn.py#L79)</span>

### predict


```python
DQNModel.predict(self, state=None)
```

----

<span style="float:right;">[[source]](https://github.com/malkoch/joatmon/blob/master/joatmon/ai/models/dqn.py#L83)</span>

### train


```python
DQNModel.train(self, batch=None, update_target=False)
```

----

<span style="float:right;">[[source]](https://github.com/malkoch/joatmon/blob/master/joatmon/ai/models/dqn.py#L114)</span>

### evaluate


```python
DQNModel.evaluate(self)
```


---
