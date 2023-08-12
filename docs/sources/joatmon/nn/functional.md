#


### wrapped_partial
```python
.wrapped_partial(
   func, *args, **kwargs
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### _check_tensor_devices
```python
._check_tensor_devices(
   *tensors: Tensor
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### _check_tensors
```python
._check_tensors(
   *tensors: Tensor
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### _get_engine
```python
._get_engine(
   *_: Union[Tensor, str]
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### _set_grad
```python
._set_grad(
   tensor: Tensor, data
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### _create_tensor
```python
._create_tensor(
   *tensors: Tensor, data, func
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### is_tensor
```python
.is_tensor(
   obj: object
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### concat_backward
```python
.concat_backward(
   gradient: Tensor, tensors: List[Tensor], axis: int = 0
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### stack_backward
```python
.stack_backward(
   gradient: Tensor, tensors: List[Tensor], axis: int = 0
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### chunk_backward
```python
.chunk_backward(
   gradient: Tensor, tensor: Tensor, chunks: int
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### view_backward
```python
.view_backward(
   gradient: Tensor, inp: Tensor
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### index_select_backward
```python
.index_select_backward(
   gradient: Tensor, inp: Tensor, index: Tensor, dim: int
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### squeeze_backward
```python
.squeeze_backward(
   gradient: Tensor, inp: Tensor
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### expand_dim_backward
```python
.expand_dim_backward(
   gradient: Tensor, inp: Tensor
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### transpose_backward
```python
.transpose_backward(
   gradient: Tensor, inp: Tensor
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### absolute_backward
```python
.absolute_backward(
   gradient: Tensor, inp: Tensor
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### around_backward
```python
.around_backward(
   gradient: Tensor, inp: Tensor
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### floor_backward
```python
.floor_backward(
   gradient: Tensor, inp: Tensor
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### ceil_backward
```python
.ceil_backward(
   gradient: Tensor, inp: Tensor
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### clip_backward
```python
.clip_backward(
   gradient: Tensor, inp: Tensor
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### negative_backward
```python
.negative_backward(
   gradient: Tensor, inp: Tensor
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### summation_backward
```python
.summation_backward(
   gradient: Tensor, inp: Tensor
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### mean_backward
```python
.mean_backward(
   gradient: Tensor, inp: Tensor
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### std_backward
```python
.std_backward(
   gradient: Tensor, inp: Tensor
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### var_backward
```python
.var_backward(
   gradient: Tensor, inp: Tensor
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### add_backward
```python
.add_backward(
   gradient: Tensor, inp1: Tensor, inp2: Tensor
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### sub_backward
```python
.sub_backward(
   gradient: Tensor, inp1: Tensor, inp2: Tensor
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### mul_backward
```python
.mul_backward(
   gradient: Tensor, inp1: Tensor, inp2: Tensor
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### div_backward
```python
.div_backward(
   gradient: Tensor, inp1: Tensor, inp2: Tensor
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### power_backward
```python
.power_backward(
   gradient: Tensor, inp: Tensor, p: int
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### clone_backward
```python
.clone_backward(
   gradient: Tensor, inp: Tensor
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### relu_backward
```python
.relu_backward(
   gradient: Tensor, inp: Tensor, alpha: float
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### sigmoid_backward
```python
.sigmoid_backward(
   gradient: Tensor, inp: Tensor, out
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### softmax_backward
```python
.softmax_backward(
   gradient: Tensor, inp: Tensor
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### tanh_backward
```python
.tanh_backward(
   gradient: Tensor, inp: Tensor, out: Tensor
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### dense_backward
```python
.dense_backward(
   gradient: Tensor, inp: Tensor, weight: Tensor, bias: Tensor
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### conv_backward
```python
.conv_backward(
   gradient: Tensor, inp: Tensor, weight: Tensor, bias: Tensor, stride: int,
   padding: Union[List[int], Tuple[int]]
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### dropout_backward
```python
.dropout_backward(
   gradient: Tensor, inp: Tensor, mask, keep_prob: float
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### batch_norm_backward
```python
.batch_norm_backward(
   gradient: Tensor, inp: Tensor, weight: Tensor, bias: Tensor, training: bool,
   **kwargs
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### max_pool_backward
```python
.max_pool_backward(
   gradient: Tensor, inp: Tensor, kernel_size: Union[List[int], Tuple[int]],
   stride: int, padding: Union[List[int], Tuple[int]], cache: dict
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### avg_pool_backward
```python
.avg_pool_backward(
   gradient: Tensor, inp: Tensor, kernel_size: Union[List[int], Tuple[int]],
   stride: int, padding: Union[List[int], Tuple[int]]
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### lstm_cell_backward
```python
.lstm_cell_backward(
   gradient, inp, all_weights, cache
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### lstm_backward
```python
.lstm_backward(
   gradient, inp, all_weights, cache
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### concat
```python
.concat(
   tensors: List[Tensor], axis: int = 0
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
   tensors: List[Tensor], axis: int = 0
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### chunk
```python
.chunk(
   tensor: Tensor, chunks: int, dim: int = 0
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### view
```python
.view(
   inp, size = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### index_select
```python
.index_select(
   inp, dim, index
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### zero
```python
.zero(
   inp
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### one
```python
.one(
   inp
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
   inp, value
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
   inp, axis = None
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### expand_dim
```python
.expand_dim(
   inp, axis = None
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
   inp, axes = None
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
   inp
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
   inp
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
   inp
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
   inp
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
   inp, min_val, max_val
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
   inp
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### summation
```python
.summation(
   inp
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
   inp
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
   inp
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
   inp
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
   inp1, inp2
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
   inp1, inp2
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
   inp1, inp2
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### div
```python
.div(
   inp1, inp2
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
   inp, p
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### clone
```python
.clone(
   inp
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### detach
```python
.detach(
   inp, inplace = True
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
   start = 0, stop = 0, step = 1, requires_grad = False, device = 'cpu', dtype = 'float32'
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
   start, end, steps, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### normal
```python
.normal(
   loc = 0.0, scale = 1.0, size = None, requires_grad = False, device = 'cpu',
   dtype = 'float32'
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### uniform
```python
.uniform(
   low = -1.0, high = 1.0, size = None, requires_grad = False, device = 'cpu',
   dtype = 'float32'
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### rand
```python
.rand(
   size, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### randint
```python
.randint(
   low = 0, high = 0, size = None, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### randn
```python
.randn(
   size, requires_grad = False, device = 'cpu', dtype = 'float32'
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
   rows, columns = None, requires_grad = False, device = 'cpu', dtype = 'float32'
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
   size, requires_grad = False, device = 'cpu', dtype = 'float32'
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
   size, fill_value, requires_grad = False, device = 'cpu', dtype = 'float32'
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
   size, requires_grad = False, device = 'cpu', dtype = 'float32'
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
   size, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### normal_like
```python
.normal_like(
   tensor, loc = 0.0, scale = 1.0, requires_grad = False, device = 'cpu',
   dtype = 'float32'
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### uniform_like
```python
.uniform_like(
   tensor, low = -1.0, high = 1.0, requires_grad = False, device = 'cpu',
   dtype = 'float32'
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### rand_like
```python
.rand_like(
   tensor, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### randint_like
```python
.randint_like(
   tensor, low = 0, high = 0, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### randn_like
```python
.randn_like(
   tensor, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### eye_like
```python
.eye_like(
   tensor, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### empty_like
```python
.empty_like(
   tensor, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### full_like
```python
.full_like(
   tensor, fill_value, requires_grad = False, device = 'cpu', dtype = 'float32'
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
   tensor, requires_grad = False, device = 'cpu', dtype = 'float32'
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
   tensor, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### from_array
```python
.from_array(
   data, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### to_array
```python
.to_array(
   inp
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### half
```python
.half(
   inp
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### single
```python
.single(
   inp
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### double
```python
.double(
   inp
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### cpu
```python
.cpu(
   inp
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### gpu
```python
.gpu(
   inp
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### relu
```python
.relu(
   inp, alpha
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### sigmoid
```python
.sigmoid(
   inp
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### softmax
```python
.softmax(
   inp
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
   inp
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### dense
```python
.dense(
   inp, weight, bias
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### conv
```python
.conv(
   inp, weight, bias, stride, padding
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### dropout
```python
.dropout(
   inp, keep_prob
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### batch_norm
```python
.batch_norm(
   inp, weight, bias, running_mean, running_var, momentum, eps, training
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### max_pool
```python
.max_pool(
   inp, kernel_size, stride, padding
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### avg_pool
```python
.avg_pool(
   inp, kernel_size, stride, padding
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### lstm_cell
```python
.lstm_cell(
   inp, all_weights
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### lstm
```python
.lstm(
   inp, all_weights
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### adam
```python
.adam(
   params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, amsgrad,
   beta1, beta2, lr, weight_decay, eps
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.

----


### rmsprop
```python
.rmsprop(
   params, grads, square_avgs, alphas, momentum_buffers, grad_avgs, momentum,
   centered, lr, weight_decay, eps
)
```

---
Remember the transaction.

Accepts a state, action, reward, next_state, terminal transaction.

# Arguments
transaction (abstract): state, action, reward, next_state, terminal transaction.
