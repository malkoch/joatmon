#


## Broadcast
```python 
Broadcast(
   inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT]
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


### .reset
```python
.reset()
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### array
```python
.array(
   arr: ArrayT
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### asarray
```python
.asarray(
   arr: ArrayT
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### arange
```python
.arange(
   start: float, stop: float, step: float
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### linspace
```python
.linspace(
   start: float, stop: float, steps: int
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### eye
```python
.eye(
   rows: int, columns: int
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### empty
```python
.empty(
   shape
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### full
```python
.full(
   shape
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### zeros
```python
.zeros(
   shape
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### ones
```python
.ones(
   shape
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### ones_like
```python
.ones_like(
   inp: ArrayT
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### zeros_like
```python
.zeros_like(
   inp: ArrayT
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### concatenate
```python
.concatenate(
   inputs, axis: AxisT = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### stack
```python
.stack(
   inputs, axis: AxisT = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### astype
```python
.astype(
   inp: Union[DataT, ArrayT], dtype: TypeT
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### copy
```python
.copy(
   inp: Union[DataT, ArrayT]
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### repeat
```python
.repeat(
   inp: Union[DataT, ArrayT], count: DataT, axis = 0
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### split
```python
.split(
   inp: ArrayT, chunks, axis = 0
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### tolist
```python
.tolist(
   inp: ArrayT
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### getitem
```python
.getitem(
   inp: ArrayT, idx: IndexT
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### take_along_axis
```python
.take_along_axis(
   inp: Union[DataT, ArrayT], indexes, axis
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### setitem
```python
.setitem(
   inp: ArrayT, idx: IndexT, value: Union[DataT, ArrayT]
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### put_along_axis
```python
.put_along_axis(
   inp: Union[DataT, ArrayT], indexes, values, axis
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### where
```python
.where(
   condition
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### indices
```python
.indices(
   dimensions
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### dim
```python
.dim(
   inp: Union[DataT, ArrayT]
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### size
```python
.size(
   inp: Union[DataT, ArrayT], axis: AxisT = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### flatten
```python
.flatten(
   inp: ArrayT
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### reshape
```python
.reshape(
   inp: ArrayT, shape
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### squeeze
```python
.squeeze(
   inp: ArrayT, axis: AxisT = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### expand_dims
```python
.expand_dims(
   inp: ArrayT, axis
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### pad
```python
.pad(
   inp: Union[DataT, ArrayT], padding, mode
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### transpose
```python
.transpose(
   inp: Union[DataT, ArrayT], axes = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### fill
```python
.fill(
   inp: ArrayT, value: DataT
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### absolute
```python
.absolute(
   inp: Union[DataT, ArrayT], *, out: Optional[ArrayT] = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### negative
```python
.negative(
   inp: Union[DataT, ArrayT], *, out: Optional[ArrayT] = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### around
```python
.around(
   inp: Union[DataT, ArrayT], *, out: Optional[ArrayT] = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### floor
```python
.floor(
   inp: Union[DataT, ArrayT], *, out: Optional[ArrayT] = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### ceil
```python
.ceil(
   inp: Union[DataT, ArrayT], *, out: Optional[ArrayT] = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### sqrt
```python
.sqrt(
   inp: Union[DataT, ArrayT], *, out: Optional[ArrayT] = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### square
```python
.square(
   inp: Union[DataT, ArrayT], *, out: Optional[ArrayT] = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### clip
```python
.clip(
   inp: Union[DataT, ArrayT], min_value: DataT, max_value: DataT, *,
   out: Optional[ArrayT] = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### exp
```python
.exp(
   inp: Union[DataT, ArrayT], *, out: Optional[ArrayT] = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### tanh
```python
.tanh(
   inp: Union[DataT, ArrayT], *, out: Optional[ArrayT] = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### sum
```python
.sum(
   inp: ArrayT, axis: AxisT = None, keepdims = False
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### mean
```python
.mean(
   inp: ArrayT, axis: AxisT = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### median
```python
.median(
   inp: ArrayT, axis: AxisT = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### var
```python
.var(
   inp: ArrayT, axis: AxisT = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### std
```python
.std(
   inp: ArrayT, axis: AxisT = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### prod
```python
.prod(
   inp: ArrayT, axis: AxisT = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### unique
```python
.unique(
   inp: ArrayT, return_counts = False
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### argmax
```python
.argmax(
   inp: ArrayT, axis: AxisT = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### argmin
```python
.argmin(
   inp: ArrayT, axis: AxisT = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### amax
```python
.amax(
   inp: ArrayT, axis: AxisT = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### maximum
```python
.maximum(
   inp: ArrayT, axis: AxisT = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### amin
```python
.amin(
   inp: ArrayT, axis: AxisT = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### minimum
```python
.minimum(
   inp: ArrayT, axis: AxisT = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### add
```python
.add(
   inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *,
   out: Optional[ArrayT] = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### sub
```python
.sub(
   inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *,
   out: Optional[ArrayT] = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### mul
```python
.mul(
   inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *,
   out: Optional[ArrayT] = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### truediv
```python
.truediv(
   inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *,
   out: Optional[ArrayT] = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### floordiv
```python
.floordiv(
   inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *,
   out: Optional[ArrayT] = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### power
```python
.power(
   inp: Union[DataT, ArrayT], p: Union[DataT, ArrayT], *,
   out: Optional[ArrayT] = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### lt
```python
.lt(
   inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *,
   out: Optional[ArrayT] = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### le
```python
.le(
   inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *,
   out: Optional[ArrayT] = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### gt
```python
.gt(
   inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *,
   out: Optional[ArrayT] = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### ge
```python
.ge(
   inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *,
   out: Optional[ArrayT] = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### eq
```python
.eq(
   inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *,
   out: Optional[ArrayT] = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### ne
```python
.ne(
   inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *,
   out: Optional[ArrayT] = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### dot
```python
.dot(
   inp1: ArrayT, inp2: ArrayT, *, out: Optional[ArrayT] = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.
