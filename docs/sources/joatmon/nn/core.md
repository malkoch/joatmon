#


## ModuleAttributeException
```python
ModuleAttributeException()
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


## RemovableHandle
```python
RemovableHandle(
   hooks_dict: Any
)
```


---
A handle which provides the capability to remove a hook.


**Methods:**


### .remove
```python
.remove()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


## Tensor
```python
Tensor(
   data = None, requires_grad = None
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


### .chunk
```python
.chunk(
   chunks, dim = 0
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .view
```python
.view(
   size
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .index_select
```python
.index_select(
   dim, index
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .zero
```python
.zero()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .one
```python
.one()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .fill
```python
.fill(
   value
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .squeeze
```python
.squeeze(
   axis = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .expand_dim
```python
.expand_dim(
   axis = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .transpose
```python
.transpose(
   axes
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .absolute
```python
.absolute()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .around
```python
.around()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .floor
```python
.floor()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .ceil
```python
.ceil()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .clip
```python
.clip(
   min_val, max_val
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .negative
```python
.negative()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .summation
```python
.summation()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .mean
```python
.mean()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .std
```python
.std()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .var
```python
.var()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .add
```python
.add(
   other
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .sub
```python
.sub(
   other
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .mul
```python
.mul(
   other
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .div
```python
.div(
   other
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .power
```python
.power(
   p
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .clone
```python
.clone()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .detach
```python
.detach(
   inplace = False
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .from_array
```python
.from_array(
   data, requires_grad = False
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .to_array
```python
.to_array()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .half
```python
.half()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .single
```python
.single()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .double
```python
.double()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .cpu
```python
.cpu()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .gpu
```python
.gpu()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .size
```python
.size(
   dim = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .dim
```python
.dim()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .shape
```python
.shape()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .ndim
```python
.ndim()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .dtype
```python
.dtype()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .device
```python
.device()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .data
```python
.data()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .is_leaf
```python
.is_leaf()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .grad
```python
.grad()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .requires_grad
```python
.requires_grad()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .retain_grad
```python
.retain_grad()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .register_hook
```python
.register_hook(
   hook
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .forward
```python
.forward()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .backward
```python
.backward(
   gradient = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


## Parameter
```python
Parameter(
   data = None, requires_grad = True
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


## Module
```python

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


### .register_parameter
```python
.register_parameter(
   name: str, param: Optional[Parameter]
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .add_module
```python
.add_module(
   name: str, module: Optional['Module']
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .parameters
```python
.parameters(
   recurse: bool = True
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .named_parameters
```python
.named_parameters(
   prefix: str = '', recurse: bool = True
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .children
```python
.children()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .named_children
```python
.named_children()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .modules
```python
.modules()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .named_modules
```python
.named_modules(
   memo: Optional[Set['Module']] = None, prefix: str = ''
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .train
```python
.train()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .eval
```python
.eval()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .zero_grad
```python
.zero_grad(
   set_to_none: bool = False
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .state_dict
```python
.state_dict()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .load_state_dict
```python
.load_state_dict()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .save_state_dict
```python
.save_state_dict()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .apply
```python
.apply(
   func: Callable[['Module'], None]
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .half
```python
.half()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .single
```python
.single()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .double
```python
.double()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .cpu
```python
.cpu()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .gpu
```python
.gpu()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .extra_repr
```python
.extra_repr()
```

---
Set the extra representation of the module

To print customized extra information, you should re-implement
this method in your own modules. Both single-line and multi-line
strings are acceptable.

----


## Loss
```python
Loss(
   size_average = None, reduce = None, reduction: str = 'mean'
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


## Sequential
```python
Sequential(
   *args: Any
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


### .forward
```python
.forward(
   inp
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


## Optimizer
```python
Optimizer(
   params, defaults
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


### .zero_grad
```python
.zero_grad(
   set_to_none: bool = False
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .step
```python
.step()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .add_param_group
```python
.add_param_group(
   param_group
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


## LRScheduler
```python
LRScheduler(
   optimizer, last_epoch = -1, verbose = False
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


### .state_dict
```python
.state_dict()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .load_state_dict
```python
.load_state_dict(
   state_dict
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .get_last_lr
```python
.get_last_lr()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .get_lr
```python
.get_lr()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .print_lr
```python
.print_lr(
   is_verbose, group, lr, epoch = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

### .step
```python
.step(
   epoch = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.
