#


### wrapped_partial
```python
.wrapped_partial(
   func, *args, **kwargs
)
```


----


### _check_tensor_devices
```python
._check_tensor_devices(
   *tensors: Tensor
)
```


----


### _check_tensors
```python
._check_tensors(
   *tensors: Tensor
)
```


----


### _get_engine
```python
._get_engine(
   *_: Union[Tensor, str]
)
```


----


### _set_grad
```python
._set_grad(
   tensor: Tensor, data
)
```


----


### _create_tensor
```python
._create_tensor(
   *tensors: Tensor, data, func
)
```


----


### is_tensor
```python
.is_tensor(
   obj: object
)
```


----


### concat_backward
```python
.concat_backward(
   gradient: Tensor, tensors: List[Tensor], axis: int = 0
)
```


----


### stack_backward
```python
.stack_backward(
   gradient: Tensor, tensors: List[Tensor], axis: int = 0
)
```


----


### chunk_backward
```python
.chunk_backward(
   gradient: Tensor, tensor: Tensor, chunks: int
)
```


----


### view_backward
```python
.view_backward(
   gradient: Tensor, inp: Tensor
)
```


----


### index_select_backward
```python
.index_select_backward(
   gradient: Tensor, inp: Tensor, index: Tensor, dim: int
)
```


----


### squeeze_backward
```python
.squeeze_backward(
   gradient: Tensor, inp: Tensor
)
```


----


### expand_dim_backward
```python
.expand_dim_backward(
   gradient: Tensor, inp: Tensor
)
```


----


### transpose_backward
```python
.transpose_backward(
   gradient: Tensor, inp: Tensor
)
```


----


### absolute_backward
```python
.absolute_backward(
   gradient: Tensor, inp: Tensor
)
```


----


### around_backward
```python
.around_backward(
   gradient: Tensor, inp: Tensor
)
```


----


### floor_backward
```python
.floor_backward(
   gradient: Tensor, inp: Tensor
)
```


----


### ceil_backward
```python
.ceil_backward(
   gradient: Tensor, inp: Tensor
)
```


----


### clip_backward
```python
.clip_backward(
   gradient: Tensor, inp: Tensor
)
```


----


### negative_backward
```python
.negative_backward(
   gradient: Tensor, inp: Tensor
)
```


----


### summation_backward
```python
.summation_backward(
   gradient: Tensor, inp: Tensor
)
```


----


### mean_backward
```python
.mean_backward(
   gradient: Tensor, inp: Tensor
)
```


----


### std_backward
```python
.std_backward(
   gradient: Tensor, inp: Tensor
)
```


----


### var_backward
```python
.var_backward(
   gradient: Tensor, inp: Tensor
)
```


----


### add_backward
```python
.add_backward(
   gradient: Tensor, inp1: Tensor, inp2: Tensor
)
```


----


### sub_backward
```python
.sub_backward(
   gradient: Tensor, inp1: Tensor, inp2: Tensor
)
```


----


### mul_backward
```python
.mul_backward(
   gradient: Tensor, inp1: Tensor, inp2: Tensor
)
```


----


### div_backward
```python
.div_backward(
   gradient: Tensor, inp1: Tensor, inp2: Tensor
)
```


----


### power_backward
```python
.power_backward(
   gradient: Tensor, inp: Tensor, p: int
)
```


----


### clone_backward
```python
.clone_backward(
   gradient: Tensor, inp: Tensor
)
```


----


### relu_backward
```python
.relu_backward(
   gradient: Tensor, inp: Tensor, alpha: float
)
```


----


### sigmoid_backward
```python
.sigmoid_backward(
   gradient: Tensor, inp: Tensor, out
)
```


----


### softmax_backward
```python
.softmax_backward(
   gradient: Tensor, inp: Tensor
)
```


----


### tanh_backward
```python
.tanh_backward(
   gradient: Tensor, inp: Tensor, out: Tensor
)
```


----


### dense_backward
```python
.dense_backward(
   gradient: Tensor, inp: Tensor, weight: Tensor, bias: Tensor
)
```


----


### conv_backward
```python
.conv_backward(
   gradient: Tensor, inp: Tensor, weight: Tensor, bias: Tensor, stride: int,
   padding: Union[List[int], Tuple[int]]
)
```


----


### dropout_backward
```python
.dropout_backward(
   gradient: Tensor, inp: Tensor, mask, keep_prob: float
)
```


----


### batch_norm_backward
```python
.batch_norm_backward(
   gradient: Tensor, inp: Tensor, weight: Tensor, bias: Tensor, training: bool,
   **kwargs
)
```


----


### max_pool_backward
```python
.max_pool_backward(
   gradient: Tensor, inp: Tensor, kernel_size: Union[List[int], Tuple[int]],
   stride: int, padding: Union[List[int], Tuple[int]], cache: dict
)
```


----


### avg_pool_backward
```python
.avg_pool_backward(
   gradient: Tensor, inp: Tensor, kernel_size: Union[List[int], Tuple[int]],
   stride: int, padding: Union[List[int], Tuple[int]]
)
```


----


### lstm_cell_backward
```python
.lstm_cell_backward(
   gradient, inp, all_weights, cache
)
```


----


### lstm_backward
```python
.lstm_backward(
   gradient, inp, all_weights, cache
)
```


----


### concat
```python
.concat(
   tensors: List[Tensor], axis: int = 0
)
```


----


### stack
```python
.stack(
   tensors: List[Tensor], axis: int = 0
)
```


----


### chunk
```python
.chunk(
   tensor: Tensor, chunks: int, dim: int = 0
)
```


----


### view
```python
.view(
   inp, size = None
)
```


----


### index_select
```python
.index_select(
   inp, dim, index
)
```


----


### zero
```python
.zero(
   inp
)
```


----


### one
```python
.one(
   inp
)
```


----


### fill
```python
.fill(
   inp, value
)
```


----


### squeeze
```python
.squeeze(
   inp, axis = None
)
```


----


### expand_dim
```python
.expand_dim(
   inp, axis = None
)
```


----


### transpose
```python
.transpose(
   inp, axes = None
)
```


----


### absolute
```python
.absolute(
   inp
)
```


----


### around
```python
.around(
   inp
)
```


----


### floor
```python
.floor(
   inp
)
```


----


### ceil
```python
.ceil(
   inp
)
```


----


### clip
```python
.clip(
   inp, min_val, max_val
)
```


----


### negative
```python
.negative(
   inp
)
```


----


### summation
```python
.summation(
   inp
)
```


----


### mean
```python
.mean(
   inp
)
```


----


### std
```python
.std(
   inp
)
```


----


### var
```python
.var(
   inp
)
```


----


### add
```python
.add(
   inp1, inp2
)
```


----


### sub
```python
.sub(
   inp1, inp2
)
```


----


### mul
```python
.mul(
   inp1, inp2
)
```


----


### div
```python
.div(
   inp1, inp2
)
```


----


### power
```python
.power(
   inp, p
)
```


----


### clone
```python
.clone(
   inp
)
```


----


### detach
```python
.detach(
   inp, inplace = True
)
```


----


### arange
```python
.arange(
   start = 0, stop = 0, step = 1, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```


----


### linspace
```python
.linspace(
   start, end, steps, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```


----


### normal
```python
.normal(
   loc = 0.0, scale = 1.0, size = None, requires_grad = False, device = 'cpu',
   dtype = 'float32'
)
```


----


### uniform
```python
.uniform(
   low = -1.0, high = 1.0, size = None, requires_grad = False, device = 'cpu',
   dtype = 'float32'
)
```


----


### rand
```python
.rand(
   size, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```


----


### randint
```python
.randint(
   low = 0, high = 0, size = None, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```


----


### randn
```python
.randn(
   size, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```


----


### eye
```python
.eye(
   rows, columns = None, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```


----


### empty
```python
.empty(
   size, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```


----


### full
```python
.full(
   size, fill_value, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```


----


### zeros
```python
.zeros(
   size, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```


----


### ones
```python
.ones(
   size, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```


----


### normal_like
```python
.normal_like(
   tensor, loc = 0.0, scale = 1.0, requires_grad = False, device = 'cpu',
   dtype = 'float32'
)
```


----


### uniform_like
```python
.uniform_like(
   tensor, low = -1.0, high = 1.0, requires_grad = False, device = 'cpu',
   dtype = 'float32'
)
```


----


### rand_like
```python
.rand_like(
   tensor, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```


----


### randint_like
```python
.randint_like(
   tensor, low = 0, high = 0, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```


----


### randn_like
```python
.randn_like(
   tensor, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```


----


### eye_like
```python
.eye_like(
   tensor, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```


----


### empty_like
```python
.empty_like(
   tensor, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```


----


### full_like
```python
.full_like(
   tensor, fill_value, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```


----


### zeros_like
```python
.zeros_like(
   tensor, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```


----


### ones_like
```python
.ones_like(
   tensor, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```


----


### from_array
```python
.from_array(
   data, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```


----


### to_array
```python
.to_array(
   inp
)
```


----


### half
```python
.half(
   inp
)
```


----


### single
```python
.single(
   inp
)
```


----


### double
```python
.double(
   inp
)
```


----


### cpu
```python
.cpu(
   inp
)
```


----


### gpu
```python
.gpu(
   inp
)
```


----


### relu
```python
.relu(
   inp, alpha
)
```


----


### sigmoid
```python
.sigmoid(
   inp
)
```


----


### softmax
```python
.softmax(
   inp
)
```


----


### tanh
```python
.tanh(
   inp
)
```


----


### dense
```python
.dense(
   inp, weight, bias
)
```


----


### conv
```python
.conv(
   inp, weight, bias, stride, padding
)
```


----


### dropout
```python
.dropout(
   inp, keep_prob
)
```


----


### batch_norm
```python
.batch_norm(
   inp, weight, bias, running_mean, running_var, momentum, eps, training
)
```


----


### max_pool
```python
.max_pool(
   inp, kernel_size, stride, padding
)
```


----


### avg_pool
```python
.avg_pool(
   inp, kernel_size, stride, padding
)
```


----


### lstm_cell
```python
.lstm_cell(
   inp, all_weights
)
```


----


### lstm
```python
.lstm(
   inp, all_weights
)
```


----


### adam
```python
.adam(
   params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, amsgrad,
   beta1, beta2, lr, weight_decay, eps
)
```


----


### rmsprop
```python
.rmsprop(
   params, grads, square_avgs, alphas, momentum_buffers, grad_avgs, momentum,
   centered, lr, weight_decay, eps
)
```

