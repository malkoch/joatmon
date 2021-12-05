## Available Callbacks

| Name                                                  | Implementation                         |
|-------------------------------------------------------|----------------------------------------|
| [GifMaker](/callback/gif-maker)                       | `joatmon.ai.callback.GifMaker`            |
| [TrainLogger](/callback/train-logger)                 | `joatmon.ai.callback.TrainLogger`         |
| [TestLogger](/callback/test-logger)                   | `joatmon.ai.callback.TestLogger`          |
| [LayerVisualizer](/callbacks/layer-visualizer)        | `joatmon.ai.callback.LayerVisualizer`     |
| [WeightLoader](/callback/weight-loader)               | `joatmon.ai.callback.WeightLoader`        |
| [EnvironmentRenderer](/callback/environment-renderer) | `joatmon.ai.callback.EnvironmentRenderer` |

---

## Common API

All callbacks share a common API. This allows you to use different callbacks.

<span style="float:right;">[[source]](https://github.com/malkoch/joatmon/blob/master/joatmon/ai/core.py#L312)</span>
### CoreModel

```python
joatmon.ai.core.CoreModel.joatmon.ai.core.CoreModel()
```


Abstract base class for all implemented nn.

Do not use this abstract base class directly
but instead use one of the concrete nn implemented.

To implement your own nn, you have to implement the following methods:

- `act`
- `replay`
- `load`
- `save`

----

<span style="float:right;">[[source]](https://github.com/malkoch/joatmon/blob/master/joatmon/ai/core.py#L330)</span>

### load


```python
CoreModel.load(self)
```



load

----

<span style="float:right;">[[source]](https://github.com/malkoch/joatmon/blob/master/joatmon/ai/core.py#L336)</span>

### save


```python
CoreModel.save(self)
```



save

----

<span style="float:right;">[[source]](https://github.com/malkoch/joatmon/blob/master/joatmon/ai/core.py#L342)</span>

### predict


```python
CoreModel.predict(self)
```



Get the action for given state.

Accepts a state and returns an abstract action.

__Arguments__

- __state__ (abstract): Current state of the environment.

__Returns__

- __action__ (abstract): Network's predicted action for given state.

----

<span style="float:right;">[[source]](https://github.com/malkoch/joatmon/blob/master/joatmon/ai/core.py#L356)</span>

### train


```python
CoreModel.train(self)
```



Train the nn with given batch.

__Arguments__

- __batch__ (abstract): Mini Batch from Experience Replay Memory.

----

<span style="float:right;">[[source]](https://github.com/malkoch/joatmon/blob/master/joatmon/ai/core.py#L365)</span>

### evaluate


```python
CoreModel.evaluate(self)
```



evaluate

