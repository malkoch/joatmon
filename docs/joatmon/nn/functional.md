#


### wrapped_partial
```python
.wrapped_partial(
   func, *args, **kwargs
)
```

---
Create a partial function and update its wrapper.


**Args**

* **func** (callable) : The original function.
* **args**  : Positional arguments to be partially applied.
* **kwargs**  : Keyword arguments to be partially applied.


**Returns**

* **callable**  : The partial function with updated wrapper.


----


### _check_tensor_devices
```python
._check_tensor_devices(
   *tensors: Tensor
)
```

---
Check if all tensors are on the same device.


**Args**

* **tensors** (Tensor) : The tensors to check.


**Raises**

* **ValueError**  : If not all tensors are on the same device.


----


### _check_tensors
```python
._check_tensors(
   *tensors: Tensor
)
```

---
Check if all arguments are tensors.


**Args**

* **tensors** (Tensor) : The arguments to check.


**Raises**

* **ValueError**  : If not all arguments are tensors.


----


### _get_engine
```python
._get_engine(
   *_: Union[Tensor, str]
)
```

---
Get the engine based on the device of the first tensor or string.


**Args**

* **_** (Union[Tensor, str]) : The tensor or string to check.


**Returns**

* **Engine**  : The engine corresponding to the device.


----


### _set_grad
```python
._set_grad(
   tensor: Tensor, data
)
```

---
Set the gradient of a tensor.


**Args**

* **tensor** (Tensor) : The tensor to set the gradient of.
* **data**  : The gradient data.


----


### _create_tensor
```python
._create_tensor(
   *tensors: Tensor, data, func
)
```

---
Create a tensor.


**Args**

* **tensors** (Tensor) : The tensors to create.
* **data**  : The data for the tensor.
* **func**  : The function to apply to the data.


**Returns**

* **Tensor**  : The created tensor.


----


### is_tensor
```python
.is_tensor(
   obj: object
)
```

---
Check if an object is a tensor.


**Args**

* **obj** (object) : The object to check.


**Returns**

* **bool**  : True if the object is a tensor, False otherwise.


----


### pad_backward
```python
.pad_backward(
   gradient: Tensor, inp: Tensor
)
```

---
Compute the backward pass of a padding operation.


**Args**

* **gradient** (Tensor) : The gradient to backpropagate.
* **inp** (Tensor) : The input tensor.


**Returns**

* **Tensor**  : The result of the backward pass.


----


### concat_backward
```python
.concat_backward(
   gradient: Tensor, tensors: List[Tensor], axis: int = 0
)
```

---
Compute the backward pass of a concatenation operation.


**Args**

* **gradient** (Tensor) : The gradient to backpropagate.
* **tensors** (List[Tensor]) : The tensors to concatenate.
* **axis** (int, optional) : The axis along which to concatenate. Defaults to 0.


**Returns**

* **Tensor**  : The result of the backward pass.


----


### stack_backward
```python
.stack_backward(
   gradient: Tensor, tensors: List[Tensor], axis: int = 0
)
```

---
Compute the backward pass of a stacking operation.


**Args**

* **gradient** (Tensor) : The gradient to backpropagate.
* **tensors** (List[Tensor]) : The tensors to stack.
* **axis** (int, optional) : The axis along which to stack. Defaults to 0.


**Returns**

* **Tensor**  : The result of the backward pass.


----


### chunk_backward
```python
.chunk_backward(
   gradient: Tensor, tensor: Tensor, chunks: int
)
```

---
Compute the backward pass of a chunking operation.


**Args**

* **gradient** (Tensor) : The gradient to backpropagate.
* **tensor** (Tensor) : The tensor to chunk.
* **chunks** (int) : The number of chunks.


**Returns**

* **Tensor**  : The result of the backward pass.


----


### view_backward
```python
.view_backward(
   gradient: Tensor, inp: Tensor
)
```

---
Compute the backward pass of a view operation.


**Args**

* **gradient** (Tensor) : The gradient to backpropagate.
* **inp** (Tensor) : The input tensor.


**Returns**

* **Tensor**  : The result of the backward pass.


----


### index_select_backward
```python
.index_select_backward(
   gradient: Tensor, inp: Tensor, index: Tensor, dim: int
)
```

---
Compute the backward pass of an index selection operation.

This function is used to perform the backward pass for index selection operation.
It takes four parameters: `gradient`, `inp`, `index`, and `dim`. The `gradient` parameter is the gradient tensor,
`inp` is the input tensor, `index` is the index tensor, and `dim` is the dimension along which to index.
The function does not return any value.


**Args**

* **gradient** (Tensor) : The gradient tensor.
* **inp** (Tensor) : The input tensor.
* **index** (Tensor) : The index tensor.
* **dim** (int) : The dimension along which to index.


**Returns**

None

----


### squeeze_backward
```python
.squeeze_backward(
   gradient: Tensor, inp: Tensor
)
```

---
Compute the backward pass of a squeeze operation.

This function is used to perform the backward pass for a squeeze operation.
It takes two parameters: `gradient` and `inp`. The `gradient` parameter is the gradient tensor,
and `inp` is the input tensor. The function does not return any value.


**Args**

* **gradient** (Tensor) : The gradient tensor.
* **inp** (Tensor) : The input tensor.


----


### expand_dim_backward
```python
.expand_dim_backward(
   gradient: Tensor, inp: Tensor
)
```

---
Compute the backward pass of an expand dimension operation.

This function is used to perform the backward pass for an expand dimension operation.
It takes two parameters: `gradient` and `inp`. The `gradient` parameter is the gradient tensor,
and `inp` is the input tensor. The function does not return any value.


**Args**

* **gradient** (Tensor) : The gradient tensor.
* **inp** (Tensor) : The input tensor.


----


### transpose_backward
```python
.transpose_backward(
   gradient: Tensor, inp: Tensor
)
```

---
Compute the backward pass of a transpose operation.

This function is used to perform the backward pass for a transpose operation.
It takes two parameters: `gradient` and `inp`. The `gradient` parameter is the gradient tensor,
and `inp` is the input tensor. The function does not return any value.


**Args**

* **gradient** (Tensor) : The gradient tensor.
* **inp** (Tensor) : The input tensor.


----


### absolute_backward
```python
.absolute_backward(
   gradient: Tensor, inp: Tensor
)
```

---
Compute the backward pass of an absolute operation.

This function is used to perform the backward pass for an absolute operation.
It takes two parameters: `gradient` and `inp`. The `gradient` parameter is the gradient tensor,
and `inp` is the input tensor. The function does not return any value.


**Args**

* **gradient** (Tensor) : The gradient tensor.
* **inp** (Tensor) : The input tensor.


----


### around_backward
```python
.around_backward(
   gradient: Tensor, inp: Tensor
)
```

---
Compute the backward pass of an around operation.

This function is used to perform the backward pass for an around operation.
It takes two parameters: `gradient` and `inp`. The `gradient` parameter is the gradient tensor,
and `inp` is the input tensor. The function does not return any value.


**Args**

* **gradient** (Tensor) : The gradient tensor.
* **inp** (Tensor) : The input tensor.


----


### floor_backward
```python
.floor_backward(
   gradient: Tensor, inp: Tensor
)
```

---
Compute the backward pass of a floor operation.

This function is used to perform the backward pass for a floor operation.
It takes two parameters: `gradient` and `inp`. The `gradient` parameter is the gradient tensor,
and `inp` is the input tensor. The function does not return any value.


**Args**

* **gradient** (Tensor) : The gradient tensor.
* **inp** (Tensor) : The input tensor.


----


### ceil_backward
```python
.ceil_backward(
   gradient: Tensor, inp: Tensor
)
```

---
Compute the backward pass of a ceil operation.

This function is used to perform the backward pass for a ceil operation.
It takes two parameters: `gradient` and `inp`. The `gradient` parameter is the gradient tensor,
and `inp` is the input tensor. The function does not return any value.


**Args**

* **gradient** (Tensor) : The gradient tensor.
* **inp** (Tensor) : The input tensor.


----


### clip_backward
```python
.clip_backward(
   gradient: Tensor, inp: Tensor
)
```

---
Compute the backward pass of a clip operation.

This function is used to perform the backward pass for a clip operation.
It takes two parameters: `gradient` and `inp`. The `gradient` parameter is the gradient tensor,
and `inp` is the input tensor. The function does not return any value.


**Args**

* **gradient** (Tensor) : The gradient tensor.
* **inp** (Tensor) : The input tensor.


----


### negative_backward
```python
.negative_backward(
   gradient: Tensor, inp: Tensor
)
```

---
Compute the backward pass of a negative operation.

This function is used to perform the backward pass for a negative operation.
It takes two parameters: `gradient` and `inp`. The `gradient` parameter is the gradient tensor,
and `inp` is the input tensor. The function does not return any value.


**Args**

* **gradient** (Tensor) : The gradient tensor.
* **inp** (Tensor) : The input tensor.


----


### log_backward
```python
.log_backward(
   gradient: Tensor, inp: Tensor
)
```

---
Compute the backward pass of a log operation.

This function is used to perform the backward pass for a log operation.
It takes two parameters: `gradient` and `inp`. The `gradient` parameter is the gradient tensor,
and `inp` is the input tensor. The function does not return any value.


**Args**

* **gradient** (Tensor) : The gradient tensor.
* **inp** (Tensor) : The input tensor.


----


### summation_backward
```python
.summation_backward(
   gradient: Tensor, inp: Tensor
)
```

---
Compute the backward pass of a summation operation.

This function is used to perform the backward pass for a summation operation.
It takes two parameters: `gradient` and `inp`. The `gradient` parameter is the gradient tensor,
and `inp` is the input tensor. The function does not return any value.


**Args**

* **gradient** (Tensor) : The gradient tensor.
* **inp** (Tensor) : The input tensor.


----


### mean_backward
```python
.mean_backward(
   gradient: Tensor, inp: Tensor
)
```

---
Compute the backward pass of a mean operation.

This function is used to perform the backward pass for a mean operation.
It takes two parameters: `gradient` and `inp`. The `gradient` parameter is the gradient tensor,
and `inp` is the input tensor. The function does not return any value.


**Args**

* **gradient** (Tensor) : The gradient tensor.
* **inp** (Tensor) : The input tensor.


----


### std_backward
```python
.std_backward(
   gradient: Tensor, inp: Tensor
)
```

---
Compute the backward pass of a standard deviation operation.

This function is used to perform the backward pass for a standard deviation operation.
It takes two parameters: `gradient` and `inp`. The `gradient` parameter is the gradient tensor,
and `inp` is the input tensor. The function does not return any value.


**Args**

* **gradient** (Tensor) : The gradient tensor.
* **inp** (Tensor) : The input tensor.


----


### var_backward
```python
.var_backward(
   gradient: Tensor, inp: Tensor
)
```

---
Compute the backward pass of a variance operation.

This function is used to perform the backward pass for a variance operation.
It takes two parameters: `gradient` and `inp`. The `gradient` parameter is the gradient tensor,
and `inp` is the input tensor. The function does not return any value.


**Args**

* **gradient** (Tensor) : The gradient tensor.
* **inp** (Tensor) : The input tensor.


----


### where_backward
```python
.where_backward(
   gradient: Tensor, condition: Tensor, tensor1: Tensor, tensor2: Tensor
)
```

---
Compute the backward pass of a where operation.

This function is used to perform the backward pass for a where operation.
It takes four parameters: `gradient`, `condition`, `tensor1`, and `tensor2`. The `gradient` parameter is the gradient tensor,
`condition` is the condition tensor, `tensor1` and `tensor2` are the tensors to choose from based on the condition.
The function does not return any value.


**Args**

* **gradient** (Tensor) : The gradient tensor.
* **condition** (Tensor) : The condition tensor.
* **tensor1** (Tensor) : The first tensor to choose from.
* **tensor2** (Tensor) : The second tensor to choose from.


----


### add_backward
```python
.add_backward(
   gradient: Tensor, inp1: Tensor, inp2: Tensor
)
```

---
Compute the backward pass of an addition operation.

This function is used to perform the backward pass for an addition operation.
It takes three parameters: `gradient`, `inp1`, and `inp2`. The `gradient` parameter is the gradient tensor,
`inp1` and `inp2` are the input tensors to be added. The function does not return any value.


**Args**

* **gradient** (Tensor) : The gradient tensor.
* **inp1** (Tensor) : The first input tensor.
* **inp2** (Tensor) : The second input tensor.


----


### sub_backward
```python
.sub_backward(
   gradient: Tensor, inp1: Tensor, inp2: Tensor
)
```

---
Compute the backward pass of a subtraction operation.

This function is used to perform the backward pass for a subtraction operation.
It takes three parameters: `gradient`, `inp1`, and `inp2`. The `gradient` parameter is the gradient tensor,
`inp1` and `inp2` are the input tensors to be subtracted. The function does not return any value.


**Args**

* **gradient** (Tensor) : The gradient tensor.
* **inp1** (Tensor) : The first input tensor.
* **inp2** (Tensor) : The second input tensor.


----


### mul_backward
```python
.mul_backward(
   gradient: Tensor, inp1: Tensor, inp2: Tensor
)
```

---
Compute the backward pass of a multiplication operation.

This function is used to perform the backward pass for a multiplication operation.
It takes three parameters: `gradient`, `inp1`, and `inp2`. The `gradient` parameter is the gradient tensor,
`inp1` and `inp2` are the input tensors to be multiplied. The function does not return any value.


**Args**

* **gradient** (Tensor) : The gradient tensor.
* **inp1** (Tensor) : The first input tensor.
* **inp2** (Tensor) : The second input tensor.


----


### div_backward
```python
.div_backward(
   gradient: Tensor, inp1: Tensor, inp2: Tensor
)
```

---
Compute the backward pass of a division operation.

This function is used to perform the backward pass for a division operation.
It takes three parameters: `gradient`, `inp1`, and `inp2`. The `gradient` parameter is the gradient tensor,
`inp1` and `inp2` are the input tensors to be divided. The function does not return any value.


**Args**

* **gradient** (Tensor) : The gradient tensor.
* **inp1** (Tensor) : The numerator input tensor.
* **inp2** (Tensor) : The denominator input tensor.


----


### power_backward
```python
.power_backward(
   gradient: Tensor, inp: Tensor, p: int
)
```

---
Compute the backward pass of a power operation.

This function is used to perform the backward pass for a power operation.
It takes three parameters: `gradient`, `inp`, and `p`. The `gradient` parameter is the gradient tensor,
`inp` is the input tensor, and `p` is the power to which the input tensor is raised. The function does not return any value.


**Args**

* **gradient** (Tensor) : The gradient tensor.
* **inp** (Tensor) : The input tensor.
* **p** (int) : The power to which the input tensor is raised.


----


### clone_backward
```python
.clone_backward(
   gradient: Tensor, inp: Tensor
)
```

---
Compute the backward pass of a clone operation.

This function is used to perform the backward pass for a clone operation.
It takes two parameters: `gradient` and `inp`. The `gradient` parameter is the gradient tensor,
and `inp` is the input tensor. The function does not return any value.


**Args**

* **gradient** (Tensor) : The gradient tensor.
* **inp** (Tensor) : The input tensor.


----


### relu_backward
```python
.relu_backward(
   gradient: Tensor, inp: Tensor, alpha: float
)
```

---
Compute the backward pass of a ReLU operation.

This function is used to perform the backward pass for a ReLU operation.
It takes three parameters: `gradient`, `inp`, and `alpha`. The `gradient` parameter is the gradient tensor,
`inp` is the input tensor, and `alpha` is the slope of the negative part of the ReLU function. The function does not return any value.


**Args**

* **gradient** (Tensor) : The gradient tensor.
* **inp** (Tensor) : The input tensor.
* **alpha** (float) : The slope of the negative part of the ReLU function.


----


### sigmoid_backward
```python
.sigmoid_backward(
   gradient: Tensor, inp: Tensor, out
)
```

---
Compute the backward pass of a sigmoid operation.

This function is used to perform the backward pass for a sigmoid operation.
It takes three parameters: `gradient`, `inp`, and `out`. The `gradient` parameter is the gradient tensor,
`inp` is the input tensor, and `out` is the output tensor from the forward pass.
The function does not return any value.


**Args**

* **gradient** (Tensor) : The gradient tensor.
* **inp** (Tensor) : The input tensor.
* **out** (Tensor) : The output tensor from the forward pass.


----


### softmax_backward
```python
.softmax_backward(
   gradient: Tensor, inp: Tensor
)
```

---
Compute the backward pass of a softmax operation.

This function is used to perform the backward pass for a softmax operation.
It takes two parameters: `gradient` and `inp`. The `gradient` parameter is the gradient tensor,
and `inp` is the input tensor.
The function does not return any value.


**Args**

* **gradient** (Tensor) : The gradient tensor.
* **inp** (Tensor) : The input tensor.


----


### tanh_backward
```python
.tanh_backward(
   gradient: Tensor, inp: Tensor, out: Tensor
)
```

---
Compute the backward pass of a tanh operation.

This function is used to perform the backward pass for a tanh operation.
It takes three parameters: `gradient`, `inp`, and `out`. The `gradient` parameter is the gradient tensor,
`inp` is the input tensor, and `out` is the output tensor from the forward pass.
The function does not return any value.


**Args**

* **gradient** (Tensor) : The gradient tensor.
* **inp** (Tensor) : The input tensor.
* **out** (Tensor) : The output tensor from the forward pass.


----


### dense_backward
```python
.dense_backward(
   gradient: Tensor, inp: Tensor, weight: Tensor, bias: Tensor
)
```

---
Compute the backward pass of a dense (fully connected) layer.

This function is used to perform the backward pass for a dense (fully connected) layer.
It takes four parameters: `gradient`, `inp`, `weight`, and `bias`. The `gradient` parameter is the gradient tensor,
`inp` is the input tensor, `weight` is the weight tensor, and `bias` is the bias tensor.
The function does not return any value.


**Args**

* **gradient** (Tensor) : The gradient tensor.
* **inp** (Tensor) : The input tensor.
* **weight** (Tensor) : The weight tensor.
* **bias** (Tensor) : The bias tensor.


----


### conv_backward
```python
.conv_backward(
   gradient: Tensor, inp: Tensor, weight: Tensor, bias: Tensor, stride: int,
   padding: Union[List[int], Tuple[int]]
)
```

---
Compute the backward pass of a convolutional layer.

This function is used to perform the backward pass for a convolutional layer.
It takes six parameters: `gradient`, `inp`, `weight`, `bias`, `stride`, and `padding`. The `gradient` parameter is the gradient tensor,
`inp` is the input tensor, `weight` is the weight tensor, `bias` is the bias tensor, `stride` is the stride of the convolution,
and `padding` is the padding of the convolution.
The function does not return any value.


**Args**

* **gradient** (Tensor) : The gradient tensor.
* **inp** (Tensor) : The input tensor.
* **weight** (Tensor) : The weight tensor.
* **bias** (Tensor) : The bias tensor.
* **stride** (int) : The stride of the convolution.
* **padding** (int) : The padding of the convolution.


----


### conv_transpose_backward
```python
.conv_transpose_backward(
   gradient: Tensor, inp: Tensor, weight: Tensor, bias: Tensor, stride: int,
   padding: Union[List[int], Tuple[int]]
)
```

---
Compute the backward pass of a transposed convolutional layer.

This function is used to perform the backward pass for a transposed convolutional layer.
It takes six parameters: `gradient`, `inp`, `weight`, `bias`, `stride`, and `padding`. The `gradient` parameter is the gradient tensor,
`inp` is the input tensor, `weight` is the weight tensor, `bias` is the bias tensor, `stride` is the stride of the convolution,
and `padding` is the padding of the convolution.
The function does not return any value.


**Args**

* **gradient** (Tensor) : The gradient tensor.
* **inp** (Tensor) : The input tensor.
* **weight** (Tensor) : The weight tensor.
* **bias** (Tensor) : The bias tensor.
* **stride** (int) : The stride of the convolution.
* **padding** (int) : The padding of the convolution.


----


### bilinear_interpolation_backward
```python
.bilinear_interpolation_backward(
   gradient: Tensor, inp: Tensor, scale_factor
)
```

---
Compute the backward pass of a bilinear interpolation operation.

This function is used to perform the backward pass for a bilinear interpolation operation.
It takes three parameters: `gradient`, `inp`, and `scale_factor`. The `gradient` parameter is the gradient tensor,
`inp` is the input tensor, and `scale_factor` is the scale factor for the interpolation.
The function does not return any value.


**Args**

* **gradient** (Tensor) : The gradient tensor.
* **inp** (Tensor) : The input tensor.
* **scale_factor** (float) : The scale factor for the interpolation.


----


### dropout_backward
```python
.dropout_backward(
   gradient: Tensor, inp: Tensor, mask, keep_prob: float
)
```

---
Compute the backward pass of a dropout operation.

This function is used to perform the backward pass for a dropout operation.
It takes four parameters: `gradient`, `inp`, `mask`, and `keep_prob`. The `gradient` parameter is the gradient tensor,
`inp` is the input tensor, `mask` is the dropout mask, and `keep_prob` is the probability of keeping a neuron active.
The function does not return any value.


**Args**

* **gradient** (Tensor) : The gradient tensor.
* **inp** (Tensor) : The input tensor.
* **mask** (Tensor) : The dropout mask.
* **keep_prob** (float) : The probability of keeping a neuron active.


----


### batch_norm_backward
```python
.batch_norm_backward(
   gradient: Tensor, inp: Tensor, weight: Tensor, bias: Tensor, training: bool,
   **kwargs
)
```

---
Compute the backward pass of a batch normalization operation.

This function is used to perform the backward pass for a batch normalization operation.
It takes five parameters: `gradient`, `inp`, `weight`, `bias`, and `training`. The `gradient` parameter is the gradient tensor,
`inp` is the input tensor, `weight` is the weight tensor, `bias` is the bias tensor, and `training` is a boolean indicating
whether the operation is performed during training or testing.
The function does not return any value.


**Args**

* **gradient** (Tensor) : The gradient tensor.
* **inp** (Tensor) : The input tensor.
* **weight** (Tensor) : The weight tensor.
* **bias** (Tensor) : The bias tensor.
* **training** (bool) : A boolean indicating whether the operation is performed during training or testing.


----


### max_pool_backward
```python
.max_pool_backward(
   gradient: Tensor, inp: Tensor, kernel_size: Union[List[int], Tuple[int]],
   stride: int, padding: Union[List[int], Tuple[int]], cache: dict
)
```

---
Compute the backward pass of a max pooling operation.

This function is used to perform the backward pass for a max pooling operation.
It takes five parameters: `gradient`, `inp`, `kernel_size`, `stride`, and `padding`. The `gradient` parameter is the gradient tensor,
`inp` is the input tensor, `kernel_size` is the size of the pooling kernel, `stride` is the stride of the pooling operation,
and `padding` is the padding of the pooling operation.
The function does not return any value.


**Args**

* **gradient** (Tensor) : The gradient tensor.
* **inp** (Tensor) : The input tensor.
* **kernel_size** (int) : The size of the pooling kernel.
* **stride** (int) : The stride of the pooling operation.
* **padding** (int) : The padding of the pooling operation.


----


### avg_pool_backward
```python
.avg_pool_backward(
   gradient: Tensor, inp: Tensor, kernel_size: Union[List[int], Tuple[int]],
   stride: int, padding: Union[List[int], Tuple[int]]
)
```

---
Compute the backward pass of an average pooling operation.

This function is used to perform the backward pass for an average pooling operation.
It takes five parameters: `gradient`, `inp`, `kernel_size`, `stride`, and `padding`. The `gradient` parameter is the gradient tensor,
`inp` is the input tensor, `kernel_size` is the size of the pooling kernel, `stride` is the stride of the pooling operation,
and `padding` is the padding of the pooling operation.
The function does not return any value.


**Args**

* **gradient** (Tensor) : The gradient tensor.
* **inp** (Tensor) : The input tensor.
* **kernel_size** (int) : The size of the pooling kernel.
* **stride** (int) : The stride of the pooling operation.
* **padding** (int) : The padding of the pooling operation.


----


### lstm_cell_backward
```python
.lstm_cell_backward(
   gradient, inp, all_weights, cache
)
```

---
Compute the backward pass of an LSTM cell.

This function is used to perform the backward pass for an LSTM cell.
It takes four parameters: `gradient`, `inp`, `all_weights`, and `cache`. The `gradient` parameter is the gradient tensor,
`inp` is the input tensor, `all_weights` is the weights of the LSTM cell, and `cache` is the cache from the forward pass.
The function does not return any value.


**Args**

* **gradient** (Tensor) : The gradient tensor.
* **inp** (Tensor) : The input tensor.
* **all_weights** (list) : The weights of the LSTM cell.
* **cache** (dict) : The cache from the forward pass.


----


### lstm_backward
```python
.lstm_backward(
   gradient, inp, all_weights, cache
)
```

---
Compute the backward pass of an LSTM layer.

This function is used to perform the backward pass for an LSTM layer.
It takes four parameters: `gradient`, `inp`, `all_weights`, and `cache`. The `gradient` parameter is the gradient tensor,
`inp` is the input tensor, `all_weights` is the weights of the LSTM layer, and `cache` is the cache from the forward pass.
The function does not return any value.


**Args**

* **gradient** (Tensor) : The gradient tensor.
* **inp** (Tensor) : The input tensor.
* **all_weights** (list) : The weights of the LSTM layer.
* **cache** (dict) : The cache from the forward pass.


----


### pad
```python
.pad(
   inp: Tensor, padding, mode = 'constant'
)
```

---
Pads the input tensor.

This function pads the input tensor based on the padding values provided.
The padding mode is set to "constant" by default.


**Args**

* **inp** (Tensor) : The input tensor to be padded.
* **padding** (tuple) : The padding values. It should be a tuple of four integers.
* **mode** (str, optional) : The padding mode. Default is "constant".


**Returns**

* **Tensor**  : The padded tensor.


----


### concat
```python
.concat(
   tensors: List[Tensor], axis: int = 0
)
```

---
Concatenates the given list of tensors along the specified axis.


**Args**

* **tensors** (List[Tensor]) : List of tensors to concatenate.
* **axis** (int, optional) : The axis along which the tensors will be concatenated. Defaults to 0.


**Returns**

* **Tensor**  : The concatenated tensor.


----


### stack
```python
.stack(
   tensors: List[Tensor], axis: int = 0
)
```

---
Stacks a list of tensors along a new axis.


**Args**

* **tensors** (List[Tensor]) : List of tensors to stack.
* **axis** (int, optional) : The axis in the result tensor along which the input tensors are stacked. Defaults to 0.


**Returns**

* **Tensor**  : A new tensor with an additional dimension.


----


### chunk
```python
.chunk(
   tensor: Tensor, chunks: int, dim: int = 0
)
```

---
Splits a tensor into a specific number of chunks along a given dimension.


**Args**

* **tensor** (Tensor) : The tensor to split.
* **chunks** (int) : The number of chunks to split the tensor into.
* **dim** (int, optional) : The dimension along which to split the tensor. Defaults to 0.


**Returns**

* A list of tensors which are a subdivision of the input tensor.


----


### view
```python
.view(
   inp, size = None
)
```

---
Returns a new tensor with the same data but different size.


**Args**

* **inp** (Tensor) : The input tensor.
* **size** (tuple, optional) : The desired size. Defaults to None.


**Returns**

* **Tensor**  : A tensor with the same data but different size.


----


### index_select
```python
.index_select(
   inp, dim, index
)
```

---
Returns a new tensor which indexes the input tensor along dimension dim using the entries in index.


**Args**

* **inp** (Tensor) : The input tensor.
* **dim** (int) : The dimension in which to index.
* **index** (Tensor) : The indices of elements to select.


**Returns**

* **Tensor**  : A tensor that contains the selected indices.


----


### zero
```python
.zero(
   inp
)
```

---
Fills the input tensor with zeros.


**Args**

* **inp** (Tensor) : The input tensor.


**Returns**

* **Tensor**  : A tensor filled with zeros.


----


### one
```python
.one(
   inp
)
```

---
Fills the input tensor with ones.


**Args**

* **inp** (Tensor) : The input tensor.


**Returns**

* **Tensor**  : A tensor filled with ones.


----


### fill
```python
.fill(
   inp, value
)
```

---
Fills the input tensor with the specified value.


**Args**

* **inp** (Tensor) : The input tensor.
* **value** (float) : The value to fill the tensor with.


**Returns**

* **Tensor**  : A tensor filled with the specified value.


----


### squeeze
```python
.squeeze(
   inp, axis = None
)
```

---
Removes dimensions of input of size 1.


**Args**

* **inp** (Tensor) : The input tensor.
* **axis** (int, optional) : The dimension to squeeze. Defaults to None.


**Returns**

* **Tensor**  : The output tensor.


----


### expand_dim
```python
.expand_dim(
   inp, axis = None
)
```

---
Expands the shape of an array.


**Args**

* **inp** (Tensor) : The input tensor.
* **axis** (int, optional) : The axis in which to expand the shape of the array. Defaults to None.


**Returns**

* **Tensor**  : The output tensor.


----


### transpose
```python
.transpose(
   inp, axes = None
)
```

---
Permute the dimensions of an array.


**Args**

* **inp** (Tensor) : The input tensor.
* **axes** (list, optional) : The list of axes to permute. Defaults to None.


**Returns**

* **Tensor**  : The output tensor.


----


### absolute
```python
.absolute(
   inp
)
```

---
Calculate the absolute value element-wise.


**Args**

* **inp** (Tensor) : The input tensor.


**Returns**

* **Tensor**  : The output tensor.


----


### around
```python
.around(
   inp
)
```

---
Evenly round to the given number of decimals.


**Args**

* **inp** (Tensor) : The input tensor.


**Returns**

* **Tensor**  : The output tensor.


----


### floor
```python
.floor(
   inp
)
```

---
Return the floor of the input, element-wise.


**Args**

* **inp** (Tensor) : The input tensor.


**Returns**

* **Tensor**  : The output tensor.


----


### ceil
```python
.ceil(
   inp
)
```

---
Return the ceiling of the input, element-wise.


**Args**

* **inp** (Tensor) : The input tensor.


**Returns**

* **Tensor**  : The output tensor.


----


### clip
```python
.clip(
   inp, min_val, max_val
)
```

---
Clip (limit) the values in an array.


**Args**

* **inp** (Tensor) : The input tensor.
* **min_val** (float) : Minimum value.
* **max_val** (float) : Maximum value.


**Returns**

* **Tensor**  : The output tensor.


----


### negative
```python
.negative(
   inp
)
```

---
Numerical negative, element-wise.


**Args**

* **inp** (Tensor) : The input tensor.


**Returns**

* **Tensor**  : The output tensor.


----


### log
```python
.log(
   inp
)
```

---
Natural logarithm, element-wise.


**Args**

* **inp** (Tensor) : The input tensor.


**Returns**

* **Tensor**  : The output tensor.


----


### summation
```python
.summation(
   inp
)
```

---
Sum of array elements over a given axis.


**Args**

* **inp** (Tensor) : The input tensor.


**Returns**

* **Tensor**  : The output tensor.


----


### mean
```python
.mean(
   inp
)
```

---
Compute the arithmetic mean along the specified axis.


**Args**

* **inp** (Tensor) : The input tensor.


**Returns**

* **Tensor**  : The output tensor.


----


### std
```python
.std(
   inp
)
```

---
Compute the standard deviation along the specified axis.


**Args**

* **inp** (Tensor) : The input tensor.


**Returns**

* **Tensor**  : The output tensor.


----


### var
```python
.var(
   inp
)
```

---
Compute the variance along the specified axis.


**Args**

* **inp** (Tensor) : The input tensor.


**Returns**

* **Tensor**  : The output tensor.


----


### where
```python
.where(
   condition, tensor1, tensor2
)
```

---
Return elements chosen from tensor1 or tensor2 depending on condition.


**Args**

* **condition** (Tensor) : The condition tensor.
* **tensor1** (Tensor) : The tensor to select elements from if condition is True.
* **tensor2** (Tensor) : The tensor to select elements from if condition is False.


**Returns**

* **Tensor**  : The output tensor.


----


### add
```python
.add(
   inp1, inp2
)
```

---
Add arguments element-wise.


**Args**

* **inp1** (Tensor) : The first input tensor.
* **inp2** (Tensor) : The second input tensor.


**Returns**

* **Tensor**  : The output tensor.


----


### sub
```python
.sub(
   inp1, inp2
)
```

---
Subtract arguments, element-wise.


**Args**

* **inp1** (Tensor) : The first input tensor.
* **inp2** (Tensor) : The second input tensor.


**Returns**

* **Tensor**  : The output tensor.


----


### mul
```python
.mul(
   inp1, inp2
)
```

---
Multiply arguments element-wise.


**Args**

* **inp1** (Tensor) : The first input tensor.
* **inp2** (Tensor) : The second input tensor.


**Returns**

* **Tensor**  : The output tensor.


----


### div
```python
.div(
   inp1, inp2
)
```

---
Divide arguments element-wise.


**Args**

* **inp1** (Tensor) : The first input tensor.
* **inp2** (Tensor) : The second input tensor.


**Returns**

* **Tensor**  : The output tensor.


----


### power
```python
.power(
   inp, p
)
```

---
First array elements raised to powers from second array, element-wise.


**Args**

* **inp** (Tensor) : The input tensor.
* **p** (float) : The exponent to raise the input tensor to.


**Returns**

* **Tensor**  : The output tensor.


----


### clone
```python
.clone(
   inp
)
```

---
Returns a copy of the input tensor.


**Args**

* **inp** (Tensor) : The input tensor.


**Returns**

* **Tensor**  : The output tensor.


----


### detach
```python
.detach(
   inp, inplace = True
)
```

---
Detaches the input tensor from the computation graph.


**Args**

* **inp** (Tensor) : The input tensor.
* **inplace** (bool, optional) : If True, detaches the tensor in-place. Defaults to True.


**Returns**

* **Tensor**  : The output tensor.


----


### arange
```python
.arange(
   start = 0, stop = 0, step = 1, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```

---
Returns a tensor with values from start to stop with a step size.


**Args**

* **start** (int, optional) : Start of interval. Defaults to 0.
* **stop** (int, optional) : End of interval. Defaults to 0.
* **step** (int, optional) : Spacing between values. Defaults to 1.
* **requires_grad** (bool, optional) : If True, the tensor requires gradient. Defaults to False.
* **device** (str, optional) : The device to create the tensor on. Defaults to 'cpu'.
* **dtype** (str, optional) : The data type of the tensor. Defaults to 'float32'.


**Returns**

* **Tensor**  : The output tensor.


----


### linspace
```python
.linspace(
   start, end, steps, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```

---
Returns a tensor of evenly spaced values over a specified range.


**Args**

* **start** (float) : The start of the range.
* **end** (float) : The end of the range.
* **steps** (int) : The number of steps.
* **requires_grad** (bool, optional) : If True, the returned tensor will require gradients. Defaults to False.
* **device** (str, optional) : The device to create the tensor on. Defaults to 'cpu'.
* **dtype** (str, optional) : The desired data type of returned tensor. Defaults to 'float32'.


**Returns**

* **Tensor**  : A tensor of size steps filled with evenly spaced values in the range from start to end.


----


### normal
```python
.normal(
   loc = 0.0, scale = 1.0, size = None, requires_grad = False, device = 'cpu',
   dtype = 'float32'
)
```

---
Returns a tensor filled with random numbers from a normal distribution.


**Args**

* **loc** (float, optional) : The mean of the normal distribution. Defaults to 0.0.
* **scale** (float, optional) : The standard deviation of the normal distribution. Defaults to 1.0.
* **size** (tuple, optional) : The shape of the output tensor. Defaults to None.
* **requires_grad** (bool, optional) : If True, the returned tensor will require gradients. Defaults to False.
* **device** (str, optional) : The device to create the tensor on. Defaults to 'cpu'.
* **dtype** (str, optional) : The desired data type of returned tensor. Defaults to 'float32'.


**Returns**

* **Tensor**  : A tensor filled with random numbers from a normal distribution.


----


### uniform
```python
.uniform(
   low = -1.0, high = 1.0, size = None, requires_grad = False, device = 'cpu',
   dtype = 'float32'
)
```

---
Returns a tensor filled with random numbers from a uniform distribution.


**Args**

* **low** (float, optional) : The lower bound of the uniform distribution. Defaults to -1.0.
* **high** (float, optional) : The upper bound of the uniform distribution. Defaults to 1.0.
* **size** (tuple, optional) : The shape of the output tensor. Defaults to None.
* **requires_grad** (bool, optional) : If True, the returned tensor will require gradients. Defaults to False.
* **device** (str, optional) : The device to create the tensor on. Defaults to 'cpu'.
* **dtype** (str, optional) : The desired data type of returned tensor. Defaults to 'float32'.


**Returns**

* **Tensor**  : A tensor filled with random numbers from a uniform distribution.


----


### rand
```python
.rand(
   size, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```

---
Returns a tensor filled with random numbers from a uniform distribution over [0, 1).


**Args**

* **size** (tuple) : The shape of the output tensor.
* **requires_grad** (bool, optional) : If True, the returned tensor will require gradients. Defaults to False.
* **device** (str, optional) : The device to create the tensor on. Defaults to 'cpu'.
* **dtype** (str, optional) : The desired data type of returned tensor. Defaults to 'float32'.


**Returns**

* **Tensor**  : A tensor filled with random numbers from a uniform distribution over [0, 1).


----


### randint
```python
.randint(
   low = 0, high = 0, size = None, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```

---
Returns a tensor filled with random integers from low (inclusive) to high (exclusive).


**Args**

* **low** (int, optional) : The lower bound of the random integers. Defaults to 0.
* **high** (int, optional) : The upper bound of the random integers. Defaults to 0.
* **size** (tuple, optional) : The shape of the output tensor. Defaults to None.
* **requires_grad** (bool, optional) : If True, the returned tensor will require gradients. Defaults to False.
* **device** (str, optional) : The device to create the tensor on. Defaults to 'cpu'.
* **dtype** (str, optional) : The desired data type of returned tensor. Defaults to 'float32'.


**Returns**

* **Tensor**  : A tensor filled with random integers from low (inclusive) to high (exclusive).


----


### randn
```python
.randn(
   size, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```

---
Returns a tensor filled with random numbers from a standard normal distribution.


**Args**

* **size** (tuple) : The shape of the output tensor.
* **requires_grad** (bool, optional) : If True, the returned tensor will require gradients. Defaults to False.
* **device** (str, optional) : The device to create the tensor on. Defaults to 'cpu'.
* **dtype** (str, optional) : The desired data type of returned tensor. Defaults to 'float32'.


**Returns**

* **Tensor**  : A tensor filled with random numbers from a standard normal distribution.


----


### eye
```python
.eye(
   rows, columns = None, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```

---
Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.


**Args**

* **rows** (int) : The number of rows in the output tensor.
* **columns** (int, optional) : The number of columns in the output tensor. If None, defaults to rows. Defaults to None.
* **requires_grad** (bool, optional) : If True, the returned tensor will require gradients. Defaults to False.
* **device** (str, optional) : The device to create the tensor on. Defaults to 'cpu'.
* **dtype** (str, optional) : The desired data type of returned tensor. Defaults to 'float32'.


**Returns**

* **Tensor**  : A 2-D tensor with ones on the diagonal and zeros elsewhere.


----


### empty
```python
.empty(
   size, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```

---
Returns a tensor filled with uninitialized data.


**Args**

* **size** (tuple) : The shape of the output tensor.
* **requires_grad** (bool, optional) : If True, the returned tensor will require gradients. Defaults to False.
* **device** (str, optional) : The device to create the tensor on. Defaults to 'cpu'.
* **dtype** (str, optional) : The desired data type of returned tensor. Defaults to 'float32'.


**Returns**

* **Tensor**  : A tensor filled with uninitialized data.


----


### full
```python
.full(
   size, fill_value, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```

---
Returns a tensor of size filled with fill_value.


**Args**

* **size** (tuple) : The shape of the output tensor.
* **fill_value** (float) : The value to fill the tensor with.
* **requires_grad** (bool, optional) : If True, the returned tensor will require gradients. Defaults to False.
* **device** (str, optional) : The device to create the tensor on. Defaults to 'cpu'.
* **dtype** (str, optional) : The desired data type of returned tensor. Defaults to 'float32'.


**Returns**

* **Tensor**  : A tensor of size filled with fill_value.


----


### zeros
```python
.zeros(
   size, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```

---
Returns a tensor filled with the scalar value 0.


**Args**

* **size** (tuple) : The shape of the output tensor.
* **requires_grad** (bool, optional) : If True, the returned tensor will require gradients. Defaults to False.
* **device** (str, optional) : The device to create the tensor on. Defaults to 'cpu'.
* **dtype** (str, optional) : The desired data type of returned tensor. Defaults to 'float32'.


**Returns**

* **Tensor**  : A tensor filled with the scalar value 0.


----


### ones
```python
.ones(
   size, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```

---
Returns a tensor filled with the scalar value 1.


**Args**

* **size** (tuple) : The shape of the output tensor.
* **requires_grad** (bool, optional) : If True, the returned tensor will require gradients. Defaults to False.
* **device** (str, optional) : The device to create the tensor on. Defaults to 'cpu'.
* **dtype** (str, optional) : The desired data type of returned tensor. Defaults to 'float32'.


**Returns**

* **Tensor**  : A tensor filled with the scalar value 1.


----


### normal_like
```python
.normal_like(
   tensor, loc = 0.0, scale = 1.0, requires_grad = False, device = 'cpu',
   dtype = 'float32'
)
```

---
Returns a tensor with the same size as input that is filled with random numbers from a normal distribution.


**Args**

* **tensor** (Tensor) : The size of tensor defined by input is used.
* **loc** (float, optional) : The mean of the normal distribution. Defaults to 0.0.
* **scale** (float, optional) : The standard deviation of the normal distribution. Defaults to 1.0.
* **requires_grad** (bool, optional) : If True, the returned tensor will require gradients. Defaults to False.
* **device** (str, optional) : The device to create the tensor on. Defaults to 'cpu'.
* **dtype** (str, optional) : The desired data type of returned tensor. Defaults to 'float32'.


**Returns**

* **Tensor**  : A tensor with the same size as input that is filled with random numbers from a normal distribution.


----


### uniform_like
```python
.uniform_like(
   tensor, low = -1.0, high = 1.0, requires_grad = False, device = 'cpu',
   dtype = 'float32'
)
```

---
Returns a tensor with the same size as input that is filled with random numbers from a uniform distribution.


**Args**

* **tensor** (Tensor) : The size of tensor defined by input is used.
* **low** (float, optional) : The lower bound of the uniform distribution. Defaults to -1.0.
* **high** (float, optional) : The upper bound of the uniform distribution. Defaults to 1.0.
* **requires_grad** (bool, optional) : If True, the returned tensor will require gradients. Defaults to False.
* **device** (str, optional) : The device to create the tensor on. Defaults to 'cpu'.
* **dtype** (str, optional) : The desired data type of returned tensor. Defaults to 'float32'.


**Returns**

* **Tensor**  : A tensor with the same size as input that is filled with random numbers from a uniform distribution.


----


### rand_like
```python
.rand_like(
   tensor, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```

---
Returns a tensor with the same size as input that is filled with random numbers from a uniform distribution over [0, 1).


**Args**

* **tensor** (Tensor) : The size of tensor defined by input is used.
* **requires_grad** (bool, optional) : If True, the returned tensor will require gradients. Defaults to False.
* **device** (str, optional) : The device to create the tensor on. Defaults to 'cpu'.
* **dtype** (str, optional) : The desired data type of returned tensor. Defaults to 'float32'.


**Returns**

* **Tensor**  : A tensor with the same size as input that is filled with random numbers from a uniform distribution over [0, 1).


----


### randint_like
```python
.randint_like(
   tensor, low = 0, high = 0, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```

---
Generates a tensor with the same shape as the input tensor, filled with random integers.


**Args**

* **tensor** (Tensor) : The input tensor.
* **low** (int, optional) : Lower bound of the random integers. Defaults to 0.
* **high** (int, optional) : Upper bound of the random integers. Defaults to 0.
* **requires_grad** (bool, optional) : If the tensor requires gradient. Defaults to False.
* **device** (str, optional) : The device to create the tensor on. Defaults to 'cpu'.
* **dtype** (str, optional) : The data type of the tensor. Defaults to 'float32'.


**Returns**

* **Tensor**  : A tensor filled with random integers.


----


### randn_like
```python
.randn_like(
   tensor, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```

---
Generates a tensor with the same shape as the input tensor, filled with random numbers from a standard normal distribution.


**Args**

* **tensor** (Tensor) : The input tensor.
* **requires_grad** (bool, optional) : If the tensor requires gradient. Defaults to False.
* **device** (str, optional) : The device to create the tensor on. Defaults to 'cpu'.
* **dtype** (str, optional) : The data type of the tensor. Defaults to 'float32'.


**Returns**

* **Tensor**  : A tensor filled with random numbers from a standard normal distribution.


----


### eye_like
```python
.eye_like(
   tensor, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```

---
Generates a 2-D tensor with ones on the diagonal and zeros elsewhere, with the same shape as the input tensor.


**Args**

* **tensor** (Tensor) : The input tensor.
* **requires_grad** (bool, optional) : If the tensor requires gradient. Defaults to False.
* **device** (str, optional) : The device to create the tensor on. Defaults to 'cpu'.
* **dtype** (str, optional) : The data type of the tensor. Defaults to 'float32'.


**Returns**

* **Tensor**  : A 2-D tensor with ones on the diagonal and zeros elsewhere.


----


### empty_like
```python
.empty_like(
   tensor, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```

---
Generates a tensor with the same shape as the input tensor, without initializing entries.


**Args**

* **tensor** (Tensor) : The input tensor.
* **requires_grad** (bool, optional) : If the tensor requires gradient. Defaults to False.
* **device** (str, optional) : The device to create the tensor on. Defaults to 'cpu'.
* **dtype** (str, optional) : The data type of the tensor. Defaults to 'float32'.


**Returns**

* **Tensor**  : A tensor with the same shape as the input tensor, without initializing entries.


----


### full_like
```python
.full_like(
   tensor, fill_value, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```

---
Generates a tensor with the same shape as the input tensor, filled with the specified value.


**Args**

* **tensor** (Tensor) : The input tensor.
* **fill_value** (float) : The value to fill the tensor with.
* **requires_grad** (bool, optional) : If the tensor requires gradient. Defaults to False.
* **device** (str, optional) : The device to create the tensor on. Defaults to 'cpu'.
* **dtype** (str, optional) : The data type of the tensor. Defaults to 'float32'.


**Returns**

* **Tensor**  : A tensor filled with the specified value.


----


### zeros_like
```python
.zeros_like(
   tensor, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```

---
Generates a tensor with the same shape as the input tensor, filled with zeros.


**Args**

* **tensor** (Tensor) : The input tensor.
* **requires_grad** (bool, optional) : If the tensor requires gradient. Defaults to False.
* **device** (str, optional) : The device to create the tensor on. Defaults to 'cpu'.
* **dtype** (str, optional) : The data type of the tensor. Defaults to 'float32'.


**Returns**

* **Tensor**  : A tensor filled with zeros.


----


### ones_like
```python
.ones_like(
   tensor, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```

---
Generates a tensor with the same shape as the input tensor, filled with ones.


**Args**

* **tensor** (Tensor) : The input tensor.
* **requires_grad** (bool, optional) : If the tensor requires gradient. Defaults to False.
* **device** (str, optional) : The device to create the tensor on. Defaults to 'cpu'.
* **dtype** (str, optional) : The data type of the tensor. Defaults to 'float32'.


**Returns**

* **Tensor**  : A tensor filled with ones.


----


### from_array
```python
.from_array(
   data, requires_grad = False, device = 'cpu', dtype = 'float32'
)
```

---
Creates a tensor from the given data.


**Args**

* **data** (array-like) : The data to create the tensor from.
* **requires_grad** (bool, optional) : If the tensor requires gradient. Defaults to False.
* **device** (str, optional) : The device to create the tensor on. Defaults to 'cpu'.
* **dtype** (str, optional) : The data type of the tensor. Defaults to 'float32'.


**Returns**

* **Tensor**  : A tensor created from the given data.


----


### to_array
```python
.to_array(
   inp
)
```

---
Converts the input tensor to a numpy array.


**Args**

* **inp** (Tensor) : The input tensor.


**Returns**

* **array**  : A numpy array with the same data as the input tensor.


----


### half
```python
.half(
   inp
)
```

---
Converts the data type of the input tensor to float16.


**Args**

* **inp** (Tensor) : The input tensor.


**Returns**

* **Tensor**  : The input tensor with data type converted to float16.


----


### single
```python
.single(
   inp
)
```

---
Converts the data type of the input tensor to float32.


**Args**

* **inp** (Tensor) : The input tensor.


**Returns**

* **Tensor**  : The input tensor with data type converted to float32.


----


### double
```python
.double(
   inp
)
```

---
Converts the data type of the input tensor to float64.


**Args**

* **inp** (Tensor) : The input tensor.


**Returns**

* **Tensor**  : The input tensor with data type converted to float64.


----


### cpu
```python
.cpu(
   inp
)
```

---
Moves the input tensor to the CPU.


**Args**

* **inp** (Tensor) : The input tensor.


**Returns**

* **Tensor**  : The input tensor moved to the CPU.


----


### gpu
```python
.gpu(
   inp
)
```

---
Moves the input tensor to the GPU.


**Args**

* **inp** (Tensor) : The input tensor.


**Returns**

* **Tensor**  : The input tensor moved to the GPU.


----


### relu
```python
.relu(
   inp, alpha = 0.0
)
```

---
Applies the rectified linear unit function element-wise on the input tensor.


**Args**

* **inp** (Tensor) : The input tensor.
* **alpha** (float, optional) : The negative slope coefficient. Defaults to 0.0.


**Returns**

* **Tensor**  : The result of applying the rectified linear unit function on the input tensor.


----


### sigmoid
```python
.sigmoid(
   inp
)
```

---
Applies the sigmoid function element-wise on the input tensor.


**Args**

* **inp** (Tensor) : The input tensor.


**Returns**

* **Tensor**  : The result of applying the sigmoid function on the input tensor.


----


### softmax
```python
.softmax(
   inp
)
```

---
Applies the softmax function element-wise on the input tensor.


**Args**

* **inp** (Tensor) : The input tensor.


**Returns**

* **Tensor**  : The result of applying the softmax function on the input tensor.


----


### tanh
```python
.tanh(
   inp
)
```

---
Applies the hyperbolic tangent function element-wise on the input tensor.


**Args**

* **inp** (Tensor) : The input tensor.


**Returns**

* **Tensor**  : The result of applying the hyperbolic tangent function on the input tensor.


----


### dense
```python
.dense(
   inp, weight, bias
)
```

---
Applies a linear transformation to the input tensor.


**Args**

* **inp** (Tensor) : The input tensor.
* **weight** (Tensor) : The weight tensor.
* **bias** (Tensor) : The bias tensor.


**Returns**

* **Tensor**  : The result of applying the linear transformation.


----


### conv
```python
.conv(
   inp, weight, bias, stride, padding
)
```

---
Applies a 2D convolution over the input tensor.


**Args**

* **inp** (Tensor) : The input tensor.
* **weight** (Tensor) : The weight tensor.
* **bias** (Tensor) : The bias tensor.
* **stride** (tuple) : The stride of the convolution.
* **padding** (tuple) : The padding added to both sides of the input.


**Returns**

* **Tensor**  : The result of applying the 2D convolution.


----


### conv_transpose
```python
.conv_transpose(
   inp, weight, bias, stride, padding
)
```

---
Applies a 2D transposed convolution over the input tensor.


**Args**

* **inp** (Tensor) : The input tensor.
* **weight** (Tensor) : The weight tensor.
* **bias** (Tensor) : The bias tensor.
* **stride** (tuple) : The stride of the convolution.
* **padding** (tuple) : The padding added to both sides of the input.


**Returns**

* **Tensor**  : The result of applying the 2D transposed convolution.


----


### bilinear_interpolation
```python
.bilinear_interpolation(
   inp, scale_factor
)
```

---
Performs bilinear interpolation on the input tensor.

# Arguments
inp (Tensor): The input tensor.
    If it's a single number, it will be applied to all dimensions.

---
# Returns
    Tensor: The interpolated tensor.

----


### dropout
```python
.dropout(
   inp, keep_prob
)
```

---
Applies dropout to the input tensor.

# Arguments
inp (Tensor): The input tensor.
keep_prob (float): The probability of keeping a neuron active.

---
# Returns
    Tensor: The tensor after applying dropout.

----


### batch_norm
```python
.batch_norm(
   inp, weight, bias, running_mean, running_var, momentum, eps, training
)
```

---
Applies batch normalization to the input tensor.

# Arguments
inp (Tensor): The input tensor.
weight (Tensor): The weight tensor for batch normalization.
bias (Tensor): The bias tensor for batch normalization.
running_mean (Tensor): The running mean tensor for batch normalization.
running_var (Tensor): The running variance tensor for batch normalization.
momentum (float): The momentum for the running mean and variance.
eps (float): A small number to avoid division by zero.
training (bool): Whether the model is in training mode.

---
# Returns
    Tensor: The tensor after applying batch normalization.

----


### max_pool
```python
.max_pool(
   inp, kernel_size, stride, padding
)
```

---
Applies max pooling to the input tensor.

# Arguments
inp (Tensor): The input tensor.
kernel_size (tuple): The size of the kernel.
stride (int): The stride of the convolution.
padding (int): The padding added to the input.

---
# Returns
    Tensor: The output tensor after applying max pooling.

----


### avg_pool
```python
.avg_pool(
   inp, kernel_size, stride, padding
)
```

---
Applies average pooling to the input tensor.

# Arguments
inp (Tensor): The input tensor.
kernel_size (tuple): The size of the kernel.
stride (int): The stride of the convolution.
padding (int): The padding added to the input.

---
# Returns
    Tensor: The output tensor after applying average pooling.

----


### lstm_cell
```python
.lstm_cell(
   inp, all_weights
)
```

---
Applies LSTM cell to the input tensor.

# Arguments
inp (Tensor): The input tensor.
all_weights (list): The weights for the LSTM cell.

---
# Returns
    Tensor: The output tensor after applying LSTM cell.

----


### lstm
```python
.lstm(
   inp, all_weights
)
```

---
Applies LSTM to the input tensor.

# Arguments
inp (Tensor): The input tensor.
all_weights (list): The weights for the LSTM.

---
# Returns
    Tensor: The output tensor after applying LSTM.

----


### adam
```python
.adam(
   params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, amsgrad,
   beta1, beta2, lr, weight_decay, eps
)
```

---
Applies Adam optimization to the parameters.

# Arguments
params (list): The parameters to be optimized.
grads (list): The gradients of the parameters.
exp_avgs (list): The exponential moving averages of gradient values.
exp_avg_sqs (list): The exponential moving averages of squared gradient values.
max_exp_avg_sqs (list): The maximum exponential moving averages of squared gradient values.
state_steps (list): The number of optimization steps.
amsgrad (bool): Whether to use the AMSGrad variant of Adam.
beta1 (float): The exponential decay rate for the first moment estimates.
beta2 (float): The exponential decay rate for the second moment estimates.
lr (float): The learning rate.
weight_decay (float): The weight decay.
eps (float): A small constant for numerical stability.

---
# Returns
    None

----


### rmsprop
```python
.rmsprop(
   params, grads, square_avgs, alphas, momentum_buffers, grad_avgs, momentum,
   centered, lr, weight_decay, eps
)
```

---
Applies the RMSProp optimization algorithm.

RMSProp is an optimizer that utilizes the magnitude of recent gradients to normalize the gradients.
We always keep a moving average over the root mean squared (hence Rms) gradients, by which we divide the current gradient.

# Arguments
params (list): List of parameters as tensors.
grads (list): List of gradients as tensors.
square_avgs (list): List of square averages as tensors.
alphas (list): List of alphas as tensors.
momentum_buffers (list): List of momentum buffers as tensors.
grad_avgs (list): List of gradient averages as tensors.
momentum (float): The momentum factor.
centered (bool): If True, gradients are normalized by the estimated variance of the gradient.
lr (float): Learning rate.
weight_decay (float): Weight decay factor.
eps (float): Term added to improve numerical stability.

---
# Returns
    None. The function operates in-place updating the params.
