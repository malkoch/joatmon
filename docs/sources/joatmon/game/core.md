#


## CoreSpace
```python
CoreSpace(
   shape = None, dtype = None
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


### .is_np_flattenable
```python
.is_np_flattenable()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .sample
```python
.sample(
   mask = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .contains
```python
.contains(
   x
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


## CoreEnv
```python

```


---
The abstract game class that is used by all agents. This class has the exact same API that OpenAI Gym uses so that integrating
with it is trivial. In contrast to the OpenAI Gym implementation, this class only defines the abstract methods without any actual implementation.

To implement your own game, you need to define the following methods:

- `seed`
- `reset`
- `step`
- `render`
- `close`

Refer to the [Gym documentation](https://gym.openai.com/docs/#environment).


**Methods:**


### .close
```python
.close()
```

---
Override in your subclass to perform any necessary cleanup.

Environments will automatically close() themselves when
garbage collected or when the program exits.

### .render
```python
.render(
   mode: str = 'human'
)
```

---
Renders the game.

The set of supported modes varies per game. (And some game do not support rendering at all.)

# Arguments
mode (str): The mode to render with. (default is 'human')

### .reset
```python
.reset(
   *args
)
```

---
Resets the state of the game and returns an initial observation.

# Returns
observation (abstract): The initial observation of the space. Initial reward is assumed to be 0.

### .seed
```python
.seed(
   seed = None
)
```

---
set the seed

### .step
```python
.step(
   action
)
```

---
Run one timestep of the game's dynamics.

Accepts an action and returns a tuple (observation, reward, done, info).

# Arguments
action (abstract): An action provided by the game.

---
# Returns
    info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).

### .goal
```python
.goal()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.
