#


## ModuleAttributeException
```python 
ModuleAttributeException()
```



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


----


## Tensor
```python 
Tensor(
   data = None, requires_grad = None
)
```




**Methods:**


### .chunk
```python
.chunk(
   chunks, dim = 0
)
```


### .view
```python
.view(
   size
)
```


### .index_select
```python
.index_select(
   dim, index
)
```


### .zero
```python
.zero()
```


### .one
```python
.one()
```


### .fill
```python
.fill(
   value
)
```


### .squeeze
```python
.squeeze(
   axis = None
)
```


### .expand_dim
```python
.expand_dim(
   axis = None
)
```


### .transpose
```python
.transpose(
   axes
)
```


### .absolute
```python
.absolute()
```


### .around
```python
.around()
```


### .floor
```python
.floor()
```


### .ceil
```python
.ceil()
```


### .clip
```python
.clip(
   min_val, max_val
)
```


### .negative
```python
.negative()
```


### .summation
```python
.summation()
```


### .mean
```python
.mean()
```


### .std
```python
.std()
```


### .var
```python
.var()
```


### .add
```python
.add(
   other
)
```


### .sub
```python
.sub(
   other
)
```


### .mul
```python
.mul(
   other
)
```


### .div
```python
.div(
   other
)
```


### .power
```python
.power(
   p
)
```


### .clone
```python
.clone()
```


### .detach
```python
.detach(
   inplace = False
)
```


### .from_array
```python
.from_array(
   data, requires_grad = False
)
```


### .to_array
```python
.to_array()
```


### .half
```python
.half()
```


### .single
```python
.single()
```


### .double
```python
.double()
```


### .cpu
```python
.cpu()
```


### .gpu
```python
.gpu()
```


### .size
```python
.size(
   dim = None
)
```


### .dim
```python
.dim()
```


### .shape
```python
.shape()
```


### .ndim
```python
.ndim()
```


### .dtype
```python
.dtype()
```


### .device
```python
.device()
```


### .data
```python
.data()
```


### .is_leaf
```python
.is_leaf()
```


### .grad
```python
.grad()
```


### .requires_grad
```python
.requires_grad()
```


### .retain_grad
```python
.retain_grad()
```


### .register_hook
```python
.register_hook(
   hook
)
```


### .forward
```python
.forward()
```


### .backward
```python
.backward(
   gradient = None
)
```


----


## Parameter
```python 
Parameter(
   data = None, requires_grad = True
)
```



----


## Module
```python 

```




**Methods:**


### .register_parameter
```python
.register_parameter(
   name: str, param: Optional[Parameter]
)
```


### .add_module
```python
.add_module(
   name: str, module: Optional['Module']
)
```


### .parameters
```python
.parameters(
   recurse: bool = True
)
```


### .named_parameters
```python
.named_parameters(
   prefix: str = '', recurse: bool = True
)
```


### .children
```python
.children()
```


### .named_children
```python
.named_children()
```


### .modules
```python
.modules()
```


### .named_modules
```python
.named_modules(
   memo: Optional[Set['Module']] = None, prefix: str = ''
)
```


### .train
```python
.train()
```


### .eval
```python
.eval()
```


### .zero_grad
```python
.zero_grad(
   set_to_none: bool = False
)
```


### .state_dict
```python
.state_dict()
```


### .load_state_dict
```python
.load_state_dict()
```


### .save_state_dict
```python
.save_state_dict()
```


### .apply
```python
.apply(
   func: Callable[['Module'], None]
)
```


### .half
```python
.half()
```


### .single
```python
.single()
```


### .double
```python
.double()
```


### .cpu
```python
.cpu()
```


### .gpu
```python
.gpu()
```


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



----


## Sequential
```python 
Sequential(
   *args: Any
)
```




**Methods:**


### .forward
```python
.forward(
   inp
)
```


----


## Optimizer
```python 
Optimizer(
   params, defaults
)
```




**Methods:**


### .zero_grad
```python
.zero_grad(
   set_to_none: bool = False
)
```


### .step
```python
.step()
```


### .add_param_group
```python
.add_param_group(
   param_group
)
```


----


## LRScheduler
```python 
LRScheduler(
   optimizer, last_epoch = -1, verbose = False
)
```




**Methods:**


### .state_dict
```python
.state_dict()
```


### .load_state_dict
```python
.load_state_dict(
   state_dict
)
```


### .get_last_lr
```python
.get_last_lr()
```


### .get_lr
```python
.get_lr()
```


### .print_lr
```python
.print_lr(
   is_verbose, group, lr, epoch = None
)
```


### .step
```python
.step(
   epoch = None
)
```

