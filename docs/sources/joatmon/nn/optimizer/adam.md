#


## Adam
```python 
Adam(
   params, lr = 0.001, betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0,
   amsgrad = False
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


### .step
```python
.step()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.
