#


## ModuleAttributeException
```python 
ModuleAttributeException()
```


---
Exception raised for errors in the module attribute.

This class inherits from the built-in `AttributeError` class and does not introduce new methods or attributes.

It is used when an attribute reference or assignment fails in the context of a module.

----


## RemovableHandle
```python 
RemovableHandle(
   hooks_dict: Any
)
```


---
A handle which provides the capability to remove a hook.

This class is used to manage hooks which are added to modules. It provides a mechanism to remove a hook by its ID.

# Attributes
id (int): The unique identifier for the hook.
next_id (int): The identifier to be assigned to the next hook.
hooks_dict_ref (weakref): A weak reference to the dictionary storing the hooks.


**Methods:**


### .remove
```python
.remove()
```

---
Removes the hook associated with this handle from the hooks dictionary.

If the hooks dictionary still exists and contains the hook associated with this handle, the hook is removed.

----


## Tensor
```python 
Tensor(
   data = None, requires_grad = None
)
```


---
This class represents a Tensor, a multi-dimensional array used in deep learning computations.


**Attributes**

* **_data**  : The actual data of the tensor.
* **_requires_grad**  : A boolean indicating whether the tensor requires gradients.
* **_grad_fn**  : The gradient function associated with the tensor.
* **_grad**  : The gradient of the tensor.
* **_name**  : The name of the tensor.
* **_backward_hooks**  : A dictionary of backward hooks.
* **_device**  : The device where the tensor is located ('cpu' or 'gpu').

---
Methods:
    backward: Backward pass of the tensor.


**Methods:**


### .chunk
```python
.chunk(
   chunks, dim = 0
)
```

---
Splits the Tensor into a specific number of chunks.


**Args**

* **chunks** (int) : The number of chunks to split the Tensor into.
* **dim** (int, optional) : The dimension along which to split the Tensor. Defaults to 0.


**Returns**

A list of Tensor objects which are chunks of the original Tensor.

### .view
```python
.view(
   size
)
```

---
Returns a new tensor with the same data but different size.


**Args**

* **size** (tuple) : The desired size.


**Returns**

A new Tensor object with the desired size.

### .index_select
```python
.index_select(
   dim, index
)
```

---
Returns a new tensor which indexes the input tensor along dimension dim using the entries in index.


**Args**

* **dim** (int) : The dimension in which to index.
* **index** (Tensor) : A tensor containing the indices to select.


**Returns**

A new Tensor which indexes the original Tensor along dimension dim using the entries in index.

### .zero
```python
.zero()
```

---
Fills the Tensor with zeros.


**Returns**

The original Tensor filled with zeros.

### .one
```python
.one()
```

---
Fills the Tensor with ones.


**Returns**

The original Tensor filled with ones.

### .fill
```python
.fill(
   value
)
```

---
Fills the Tensor with the specified value.


**Args**

* **value**  : The value to fill the Tensor with.


**Returns**

The original Tensor filled with the specified value.

### .squeeze
```python
.squeeze(
   axis = None
)
```

---
Removes dimensions of size one from the Tensor.


**Args**

* **axis** (int, optional) : The specific dimension to remove. If None, all dimensions of size one will be removed.


**Returns**

A new Tensor with the dimensions of size one removed.

### .expand_dim
```python
.expand_dim(
   axis = None
)
```

---
Expands the dimensions of the Tensor.


**Args**

* **axis** (int, optional) : The dimension to expand. If None, all dimensions will be expanded.


**Returns**

A new Tensor with the dimensions expanded.

### .transpose
```python
.transpose(
   axes
)
```

---
Transposes the Tensor.


**Args**

* **axes** (tuple) : The axes along which to transpose the Tensor.


**Returns**

A new Tensor that is the transpose of the original Tensor.

### .absolute
```python
.absolute()
```

---
Computes the absolute values of the Tensor.


**Returns**

A new Tensor that is the absolute value of the original Tensor.

### .around
```python
.around()
```

---
Rounds the Tensor to the nearest integer.


**Returns**

A new Tensor that is the rounded version of the original Tensor.

### .floor
```python
.floor()
```

---
Rounds the Tensor down to the nearest integer.


**Returns**

A new Tensor that is the floor of the original Tensor.

### .ceil
```python
.ceil()
```

---
Rounds the Tensor up to the nearest integer.


**Returns**

A new Tensor that is the ceiling of the original Tensor.

### .clip
```python
.clip(
   min_val, max_val
)
```

---
Clips the Tensor to be within a specified range.


**Args**

* **min_val**  : The minimum value for the Tensor.
* **max_val**  : The maximum value for the Tensor.


**Returns**

A new Tensor that is the original Tensor clipped to be within the specified range.

### .negative
```python
.negative()
```

---
Computes the negative of the Tensor.


**Returns**

A new Tensor that is the negative of the original Tensor.

### .log
```python
.log()
```

---
Computes the natural logarithm of the Tensor.


**Returns**

A new Tensor that is the natural logarithm of the original Tensor.

### .summation
```python
.summation()
```

---
Computes the sum of all elements in the Tensor.


**Returns**

A new Tensor that is the sum of all elements in the original Tensor.

### .mean
```python
.mean()
```

---
Computes the mean of all elements in the Tensor.


**Returns**

A new Tensor that is the mean of all elements in the original Tensor.

### .std
```python
.std()
```

---
Computes the standard deviation of all elements in the Tensor.


**Returns**

A new Tensor that is the standard deviation of all elements in the original Tensor.

### .var
```python
.var()
```

---
Computes the variance of all elements in the Tensor.


**Returns**

A new Tensor that is the variance of all elements in the original Tensor.

### .add
```python
.add(
   other
)
```

---
Adds another Tensor to the current Tensor.


**Args**

* **other**  : The other Tensor to add.


**Returns**

A new Tensor that is the result of the addition operation.

### .sub
```python
.sub(
   other
)
```

---
Subtracts another Tensor from the current Tensor.


**Args**

* **other**  : The other Tensor to subtract.


**Returns**

A new Tensor that is the result of the subtraction operation.

### .mul
```python
.mul(
   other
)
```

---
Multiplies the current Tensor by another Tensor.


**Args**

* **other**  : The other Tensor to multiply.


**Returns**

A new Tensor that is the result of the multiplication operation.

### .div
```python
.div(
   other
)
```

---
Divides the current Tensor by another Tensor.


**Args**

* **other**  : The other Tensor to divide.


**Returns**

A new Tensor that is the result of the division operation.

### .power
```python
.power(
   p
)
```

---
Raises the current Tensor to the power of p.


**Args**

* **p**  : The power to raise the Tensor to.


**Returns**

A new Tensor that is the result of the power operation.

### .clone
```python
.clone()
```

---
Creates a copy of the current Tensor.


**Returns**

A new Tensor that is a copy of the current Tensor.

### .detach
```python
.detach(
   inplace = False
)
```

---
Detaches the Tensor from the computation graph.


**Args**

* **inplace** (bool, optional) : If True, the operation is performed in-place. Defaults to False.


**Returns**

The detached Tensor.

### .from_array
```python
.from_array(
   data, requires_grad = False
)
```

---
Creates a Tensor from a numpy array.


**Args**

* **data**  : The numpy array to convert into a Tensor.
* **requires_grad** (bool, optional) : If True, the Tensor will require gradient computation. Defaults to False.


**Returns**

A new Tensor created from the numpy array.

### .to_array
```python
.to_array()
```

---
Converts the Tensor into a numpy array.


**Returns**

A numpy array that is a copy of the Tensor data.

### .half
```python
.half()
```

---
Converts the Tensor to half precision.


**Returns**

A new Tensor that is the half precision version of the original Tensor.

### .single
```python
.single()
```

---
Converts the Tensor to single precision.


**Returns**

A new Tensor that is the single precision version of the original Tensor.

### .double
```python
.double()
```

---
Converts the Tensor to double precision.


**Returns**

A new Tensor that is the double precision version of the original Tensor.

### .cpu
```python
.cpu()
```

---
Moves the Tensor to the CPU.


**Returns**

The Tensor after it has been moved to the CPU.

### .gpu
```python
.gpu()
```

---
Moves the Tensor to the GPU.


**Returns**

The Tensor after it has been moved to the GPU.

### .size
```python
.size(
   dim = None
)
```

---
Returns the size of the Tensor.


**Args**

* **dim** (int, optional) : If specified, the size of the specific dimension is returned. Otherwise, the size of all dimensions is returned.


**Returns**

The size of the Tensor.

### .dim
```python
.dim()
```

---
Returns the number of dimensions of the Tensor.


**Returns**

The number of dimensions of the Tensor.

### .shape
```python
.shape()
```

---
Returns the shape of the Tensor.


**Returns**

The shape of the Tensor.

### .ndim
```python
.ndim()
```

---
Returns the number of dimensions of the Tensor.


**Returns**

The number of dimensions of the Tensor.

### .dtype
```python
.dtype()
```

---
Returns the data type of the Tensor's underlying data.

# Returns
The data type of the Tensor's underlying data.

### .device
```python
.device()
```

---
Returns the device where the Tensor is located.

# Returns
A string representing the device where the Tensor is located. It could be 'cpu' or 'gpu'.

### .data
```python
.data()
```

---
Returns the underlying data of the Tensor.

# Returns
The underlying data of the Tensor.

### .is_leaf
```python
.is_leaf()
```

---
Checks if the Tensor is a leaf node.

A Tensor is considered a leaf if it was not the result of an operation.
That is, if it was read from data or if it is a constant.

# Returns
True if the Tensor is a leaf node, False otherwise.

### .grad
```python
.grad()
```

---
Returns the gradient of the Tensor.

The gradient is computed with respect to some scalar value.

# Returns
The gradient of the Tensor.

### .requires_grad
```python
.requires_grad()
```

---
Checks if the Tensor requires gradient computation.

# Returns
True if the Tensor requires gradient computation, False otherwise.

### .retain_grad
```python
.retain_grad()
```

---
Allows a non-leaf Tensor to retain its gradient.

Normally, only leaf Tensors (those not resulting from an operation) will have their gradients retained.
This method allows non-leaf Tensors to retain their gradients.

### .register_hook
```python
.register_hook(
   hook
)
```

---
Registers a backward hook.

Backward hooks are functions that are executed every time a backward operation is performed.

# Arguments
hook (function): The backward hook function to register.

### .forward
```python
.forward()
```

---
Performs the forward pass of the Tensor.

This method should be overridden by all subclasses.

### .backward
```python
.backward(
   gradient = None
)
```

---
Performs the backward pass of the Tensor.

Computes the gradient of the Tensor with respect to some scalar value.

# Arguments
                             If None, a Tensor of ones, with the same shape as the current Tensor, is used.

----


## Parameter
```python 
Parameter(
   data = None, requires_grad = True
)
```


---
A Parameter is a kind of Tensor that is to be considered a module parameter.

Parameters are Tensor subclasses, that have a very special property when used with Module s - when they’re assigned as Module attributes they are automatically added to the list of its parameters, and will appear in parameters() iterator. Assigning a Tensor doesn’t have such effect. This is because one might want to cache some temporary state (more on this later) in the model. If there was no such class as Parameter, these temporaries would get registered too.

Another difference is that parameters can't be volatile and that they require gradient by default.


**Args**

* **data** (Tensor, optional) : Parameter data. Default: None
* **requires_grad** (bool, optional) : If the parameter requires gradient. Default: `True`


----


## Module
```python 

```


---
Base class for all neural network modules.

Your models should also subclass this class.

Modules can also contain other Modules, allowing to nest them in
a tree structure. You can assign the submodules as regular attributes.

Subclassed Modules should initialize the parent class with the arguments
they receive.


**Attributes**

* **dump_patches** (bool) : If True, the model will dump its state_dict() after each forward pass.
* **_version** (int) : The version of the model.
* **training** (bool) : If True, the model is in training mode. If False, the model is in evaluation mode.



**Methods:**


### .register_parameter
```python
.register_parameter(
   name: str, param: Optional[Parameter]
)
```

---
Adds a parameter to the module.

The parameter can be accessed as an attribute using the given name.


**Args**

* **name** (str) : name of the parameter. The string should be a valid attribute name.
* **param** (Parameter, optional) : parameter to be added to the module. The parameter should be an instance of the Parameter class.


### .add_module
```python
.add_module(
   name: str, module: Optional['Module']
)
```

---
Adds a child module to the current module.

The module can be accessed as an attribute using the given name.


**Args**

* **name** (str) : name of the child module. The string should be a valid attribute name.
* **module** (Module, optional) : child module to be added to the module. The module should be an instance of the Module class.


### .state_dict
```python
.state_dict(
   *, destination = None, prefix = ''
)
```

---
Returns a dictionary representing the state of the module.

If a destination is provided, it is used. Otherwise, a new dictionary is created.


**Args**

* **destination** (dict, optional) : the dictionary to which the state should be saved. If None, a new dictionary is created.
* **prefix** (str, optional) : the prefix to be used for the keys in the state dictionary.


**Returns**

* **dict**  : a dictionary representing the state of the module.


### .load_state_dict
```python
.load_state_dict(
   state_dict: Mapping[str, Any]
)
```

---
Loads the state of the module from the state_dict.


**Args**

* **state_dict** (dict) : the dictionary from which the state should be loaded.


### .parameters
```python
.parameters(
   recurse: bool = True
)
```

---
Returns an iterator over module parameters.

This is typically passed to an optimizer.


**Args**

* **recurse** (bool) : If True, then yields parameters of this module and all submodules.
                Otherwise, yields only parameters that are direct children of this module.


**Yields**

* **Parameter**  : Module parameter.


### .named_parameters
```python
.named_parameters(
   prefix: str = '', recurse: bool = True
)
```

---
Returns an iterator over module parameters, yielding both the name of the parameter
as well as the parameter itself.


**Args**

* **prefix** (str, optional) : Prefix for each parameter name. Defaults to ''.
* **recurse** (bool, optional) : If True, includes parameters of this module and all submodules.
                          If False, includes only parameters that are direct members of this module.
                          Defaults to True.


**Yields**

* An iterator over name, parameter pairs.


### .children
```python
.children()
```

---
Returns an iterator over immediate child modules.


**Yields**

* An iterator over child modules.


### .named_children
```python
.named_children()
```

---
Returns an iterator over immediate child modules, yielding both the name of the child module
as well as the module itself.


**Yields**

* An iterator over name, module pairs.


### .modules
```python
.modules()
```

---
Returns an iterator over all modules in the current module.


**Yields**

* An iterator over all modules.


### .named_modules
```python
.named_modules(
   memo: Optional[Set['Module']] = None, prefix: str = ''
)
```

---
Returns an iterator over all modules in the current module, yielding both the name of the module
as well as the module itself.


**Args**

* **memo** (Optional[Set['Module']], optional) : Set of modules already processed. Defaults to None.
* **prefix** (str, optional) : Prefix for each module name. Defaults to ''.


**Yields**

* An iterator over name, module pairs.


### .train
```python
.train()
```

---
Sets the module in training mode.


**Returns**

* **self**  : The current module.


### .eval
```python
.eval()
```

---
Sets the module in evaluation mode.


**Returns**

* **self**  : The current module.


### .zero_grad
```python
.zero_grad(
   set_to_none: bool = False
)
```

---
Sets the gradients of all parameters to zero.


**Args**

* **set_to_none** (bool, optional) : If True, instead of setting to zero, sets the gradients to None. Defaults to False.


### .apply
```python
.apply(
   func: Callable[['Module'], None]
)
```

---
Applies a function on all modules (self and children).


**Args**

* **func** (Callable[['Module'], None]) : A function that will be applied to all modules.


### .half
```python
.half()
```

---
Casts all parameters and buffers to half precision.


**Returns**

* **self**  : The current module.


### .single
```python
.single()
```

---
Casts all parameters and buffers to single precision.


**Returns**

* **self**  : The current module.


### .double
```python
.double()
```

---
Casts all parameters and buffers to double precision.


**Returns**

* **self**  : The current module.


### .cpu
```python
.cpu()
```

---
Moves all parameters and buffers to the CPU.


**Returns**

* **self**  : The current module.


### .gpu
```python
.gpu()
```

---
Moves all parameters and buffers to the GPU.


**Returns**

* **self**  : The current module.


### .extra_repr
```python
.extra_repr()
```

---
Sets the extra representation of the module.
To print customized extra information, you should re-implement this method in your own modules.
Both single-line and multi-line strings are acceptable.


**Returns**

* **str**  : Extra representation string (empty by default).


----


## Sequential
```python 
Sequential(
   *args
)
```


---
A sequential container. Modules will be added to it in the order they are passed in the constructor.
Alternatively, an ordered dict of modules can also be passed in.


**Args**

* **args** (optional) : an ordered list of modules.



**Methods:**


### .forward
```python
.forward(
   inp
)
```

---
Defines the computation performed at every call.


**Args**

* **inp**  : the input to the forward function.


**Returns**

The output of the forward function.

----


## Loss
```python 
Loss(
   size_average = None, reduce = None, reduction: str = 'mean'
)
```


---
This class represents a Loss module which is a subclass of the Module class.
It is used to compute the loss value during the training of a neural network model.

The class uses the concept of Deep Deterministic Policy Gradient for loss computation.


**Attributes**

* **reduction** (str) : Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                 'none': no reduction will be applied.
                 'mean': the sum of the output will be divided by the number of elements in the output.
                 'sum': the output will be summed.


**Args**

* **size_average** (bool, optional) : Deprecated (see reduction). By default, the losses are averaged over each loss element in the batch.
                                Note that for some losses, there are multiple elements per sample.
                                If the field size_average is set to False, the losses are instead summed for each minibatch.
                                Ignored when reduce is False. Default: True
* **reduce** (bool, optional) : Deprecated (see reduction). By default, the losses are averaged or summed over observations for each minibatch depending
                          on size_average. When reduce is False, returns a loss per batch element instead and ignores size_average.
                          Default: True
* **reduction** (str, optional) : Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                            'none': no reduction will be applied.
                            'mean': the sum of the output will be divided by the number of elements in the output.
                            'sum': the output will be summed.
                            Default: 'mean'


----


## Optimizer
```python 
Optimizer(
   params, defaults
)
```


---
Base class for all optimizers.

Optimizers are used to update the parameters of a model in order to minimize the loss function.


**Args**

* **params** (iterable) : An iterable of Parameters that define what Tensors should be optimized.
* **defaults** (dict) : A dictionary containing default values of optimization options (like learning rate, weight decay, etc).


**Attributes**

* **defaults** (dict) : The default optimization options.
* **state** (dict) : A dictionary that holds current optimization state. Its content
    differs between optimizer classes.
* **param_groups** (list) : A list of parameter groups. Each group is a dictionary that
    holds parameters and their corresponding optimization options.



**Methods:**


### .zero_grad
```python
.zero_grad(
   set_to_none: bool = False
)
```

---
Clears the gradients of all optimized Tensors.


**Args**

* **set_to_none** (bool, optional) : Instead of filling with zero, sets the gradients to None.
    This will in general have lower memory footprint. Defaults to False.


### .step
```python
.step()
```

---
Performs a single optimization step (parameter update).

Should be overridden by all subclasses.


**Note**

It is recommended to use `torch.no_grad()` on the enclosing scope to disable
gradient computation for performance.


**Raises**

* **NotImplementedError**  : This method needs to be implemented in subclasses.


### .add_param_group
```python
.add_param_group(
   param_group
)
```

---
Adds a parameter group to the optimizer’s param_groups.

This can be useful when fine tuning a pre-trained network as frozen layers can be made
trainable and added to the optimizer as training progresses.


**Args**

* **param_group** (dict) : Specifies what Tensors should be optimized along with group-specific optimization options.


----


## LRScheduler
```python 
LRScheduler(
   optimizer, last_epoch = -1, verbose = False
)
```


---
Base class for learning rate schedulers.

This class provides the basic structure for implementing different learning rate scheduling policies.
Learning rate schedulers dynamically adjust the learning rate of the optimizer during the training process,
which can lead to improved model performance and robustness.


**Args**

* **optimizer** (Optimizer) : The optimizer for which the learning rate will be scheduled.
* **last_epoch** (int, optional) : The index of the last epoch. Default: -1.
* **verbose** (bool, optional) : If True, prints a message to stdout for each update. Default: False.



**Methods:**


### .state_dict
```python
.state_dict()
```

---
Returns the state of the scheduler as a dictionary.

It contains all items necessary to keep track of the scheduler's state.
Note that it does not contain the state of the optimizer.


**Returns**

* **dict**  : The state of the scheduler.


### .load_state_dict
```python
.load_state_dict(
   state_dict
)
```

---
Loads the schedulers state.


**Args**

* **state_dict** (dict) : Scheduler state. Should be an object returned from a call to `state_dict`.


### .get_last_lr
```python
.get_last_lr()
```

---
Returns last computed learning rate by scheduler.


**Returns**

* **list**  : Last computed learning rate by scheduler.


### .get_lr
```python
.get_lr()
```

---
Computes the learning rate at each step. Needs to be implemented by subclasses.


**Raises**

* **NotImplementedError**  : This method needs to be implemented in subclasses.


### .print_lr
```python
.print_lr(
   is_verbose, group, lr, epoch = None
)
```

---
Prints the learning rate of a specific parameter group.


**Args**

* **is_verbose** (bool) : If True, prints a message to stdout for each update.
* **group** (int) : Index of the parameter group.
* **lr** (float) : Learning rate of the parameter group.
* **epoch** (int, optional) : Current epoch number.


### .step
```python
.step(
   epoch = None
)
```

---
Updates the learning rate of the optimizer.


**Args**

* **epoch** (int, optional) : Current epoch number. Default: None.

