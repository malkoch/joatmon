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


### .on_action_begin
```python
.on_action_begin(
   *args, **kwargs
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .on_action_end
```python
.on_action_end(
   *args, **kwargs
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .on_agent_begin
```python
.on_agent_begin(
   *args, **kwargs
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .on_agent_end
```python
.on_agent_end(
   *args, **kwargs
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .on_episode_begin
```python
.on_episode_begin(
   *args, **kwargs
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .on_episode_end
```python
.on_episode_end(
   *args, **kwargs
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .on_replay_begin
```python
.on_replay_begin(
   *args, **kwargs
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .on_replay_end
```python
.on_replay_end(
   *args, **kwargs
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


## Loader
```python
Loader(
   model, run_path, interval
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


### .on_agent_begin
```python
.on_agent_begin(
   *args, **kwargs
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .on_agent_end
```python
.on_agent_end(
   *args, **kwargs
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .on_episode_end
```python
.on_episode_end(
   *args, **kwargs
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


## Renderer
```python
Renderer(
   environment
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


### .on_action_end
```python
.on_action_end(
   *args, **kwargs
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .on_episode_begin
```python
.on_episode_begin(
   *args, **kwargs
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


## TrainLogger
```python
TrainLogger(
   run_path, interval
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


### .on_agent_begin
```python
.on_agent_begin(
   *args, **kwargs
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .on_episode_begin
```python
.on_episode_begin(
   *args, **kwargs
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .on_episode_end
```python
.on_episode_end(
   *args, **kwargs
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .on_replay_end
```python
.on_replay_end(
   *args, **kwargs
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


## ValidationLogger
```python
ValidationLogger(
   run_path, interval
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


### .on_agent_begin
```python
.on_agent_begin(
   *args, **kwargs
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .on_episode_begin
```python
.on_episode_begin(
   *args, **kwargs
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .on_episode_end
```python
.on_episode_end(
   *args, **kwargs
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


## Visualizer
```python
Visualizer(
   model, predicate = lambdax: True
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


### .on_action_begin
```python
.on_action_begin(
   *args, **kwargs
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


## LivePlotter
```python
LivePlotter()
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


### .live_plotter
```python
.live_plotter(
   y_num, size
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.
