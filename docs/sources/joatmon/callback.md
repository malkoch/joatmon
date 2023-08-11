#


## CoreCallback
```python 

```


---
Abstract base class for all implemented callback.

Do not use this abstract base class directly but instead use one of the concrete callback implemented.

To implement your own callback, you have to implement the following methods:

- `on_action_begin`
- `on_action_end`
- `on_replay_begin`
- `on_replay_end`
- `on_episode_begin`
- `on_episode_end`
- `on_agent_begin`
- `on_agent_end`


**Methods:**


### .on_agent_begin
```python
.on_agent_begin(
   *args, **kwargs
)
```

---
Called at beginning of each agent play

### .on_agent_end
```python
.on_agent_end(
   *args, **kwargs
)
```

---
Called at end of each agent play

### .on_episode_begin
```python
.on_episode_begin(
   *args, **kwargs
)
```

---
Called at beginning of each game episode

### .on_episode_end
```python
.on_episode_end(
   *args, **kwargs
)
```

---
Called at end of each game episode

### .on_action_begin
```python
.on_action_begin(
   *args, **kwargs
)
```

---
Called at beginning of each agent action

### .on_action_end
```python
.on_action_end(
   *args, **kwargs
)
```

---
Called at end of each agent action

### .on_replay_begin
```python
.on_replay_begin(
   *args, **kwargs
)
```

---
Called at beginning of each nn replay

### .on_replay_end
```python
.on_replay_end(
   *args, **kwargs
)
```

---
Called at end of each nn replay

----


## CallbackList
```python 
CallbackList(
   callbacks
)
```




**Methods:**


### .on_action_begin
```python
.on_action_begin(
   *args, **kwargs
)
```


### .on_action_end
```python
.on_action_end(
   *args, **kwargs
)
```


### .on_agent_begin
```python
.on_agent_begin(
   *args, **kwargs
)
```


### .on_agent_end
```python
.on_agent_end(
   *args, **kwargs
)
```


### .on_episode_begin
```python
.on_episode_begin(
   *args, **kwargs
)
```


### .on_episode_end
```python
.on_episode_end(
   *args, **kwargs
)
```


### .on_replay_begin
```python
.on_replay_begin(
   *args, **kwargs
)
```


### .on_replay_end
```python
.on_replay_end(
   *args, **kwargs
)
```


----


## Loader
```python 
Loader(
   model, run_path, interval
)
```




**Methods:**


### .on_agent_begin
```python
.on_agent_begin(
   *args, **kwargs
)
```


### .on_agent_end
```python
.on_agent_end(
   *args, **kwargs
)
```


### .on_episode_end
```python
.on_episode_end(
   *args, **kwargs
)
```


----


## Renderer
```python 
Renderer(
   environment
)
```




**Methods:**


### .on_action_end
```python
.on_action_end(
   *args, **kwargs
)
```


### .on_episode_begin
```python
.on_episode_begin(
   *args, **kwargs
)
```


----


## TrainLogger
```python 
TrainLogger(
   run_path, interval
)
```




**Methods:**


### .on_agent_begin
```python
.on_agent_begin(
   *args, **kwargs
)
```


### .on_episode_begin
```python
.on_episode_begin(
   *args, **kwargs
)
```


### .on_episode_end
```python
.on_episode_end(
   *args, **kwargs
)
```


### .on_replay_end
```python
.on_replay_end(
   *args, **kwargs
)
```


----


## ValidationLogger
```python 
ValidationLogger(
   run_path, interval
)
```




**Methods:**


### .on_agent_begin
```python
.on_agent_begin(
   *args, **kwargs
)
```


### .on_episode_begin
```python
.on_episode_begin(
   *args, **kwargs
)
```


### .on_episode_end
```python
.on_episode_end(
   *args, **kwargs
)
```


----


## Visualizer
```python 
Visualizer(
   model, predicate = lambdax: True
)
```




**Methods:**


### .on_action_begin
```python
.on_action_begin(
   *args, **kwargs
)
```


----


## LivePlotter
```python 
LivePlotter()
```




**Methods:**


### .live_plotter
```python
.live_plotter(
   y_num, size
)
```

