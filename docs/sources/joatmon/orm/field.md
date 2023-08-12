#


## Field
```python
Field(
   dtype: typing.Union[type, typing.List, typing.Tuple], nullable: bool = True,
   default = None, primary: bool = False, encrypt: bool = False, hash_: bool = False,
   resource: str = None, regex: str = None, fields: dict = None
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

----


### get_converter
```python
.get_converter(
   field: Field
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.
