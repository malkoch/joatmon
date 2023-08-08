#


## Broadcast
```python 
Broadcast(
   inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT]
)
```




**Methods:**


### .reset
```python
.reset()
```


----


### array
```python
.array(
   arr: ArrayT
)
```


----


### asarray
```python
.asarray(
   arr: ArrayT
)
```


----


### arange
```python
.arange(
   start: float, stop: float, step: float
)
```


----


### linspace
```python
.linspace(
   start: float, stop: float, steps: int
)
```


----


### eye
```python
.eye(
   rows: int, columns: int
)
```


----


### empty
```python
.empty(
   shape
)
```


----


### full
```python
.full(
   shape
)
```


----


### zeros
```python
.zeros(
   shape
)
```


----


### ones
```python
.ones(
   shape
)
```


----


### ones_like
```python
.ones_like(
   inp: ArrayT
)
```


----


### zeros_like
```python
.zeros_like(
   inp: ArrayT
)
```


----


### concatenate
```python
.concatenate(
   inputs, axis: AxisT = None
)
```


----


### stack
```python
.stack(
   inputs, axis: AxisT = None
)
```


----


### astype
```python
.astype(
   inp: Union[DataT, ArrayT], dtype: TypeT
)
```


----


### copy
```python
.copy(
   inp: Union[DataT, ArrayT]
)
```


----


### repeat
```python
.repeat(
   inp: Union[DataT, ArrayT], count: DataT, axis = 0
)
```


----


### split
```python
.split(
   inp: ArrayT, chunks, axis = 0
)
```


----


### tolist
```python
.tolist(
   inp: ArrayT
)
```


----


### getitem
```python
.getitem(
   inp: ArrayT, idx: IndexT
)
```


----


### take_along_axis
```python
.take_along_axis(
   inp: Union[DataT, ArrayT], indexes, axis
)
```


----


### setitem
```python
.setitem(
   inp: ArrayT, idx: IndexT, value: Union[DataT, ArrayT]
)
```


----


### put_along_axis
```python
.put_along_axis(
   inp: Union[DataT, ArrayT], indexes, values, axis
)
```


----


### where
```python
.where(
   condition
)
```


----


### indices
```python
.indices(
   dimensions
)
```


----


### dim
```python
.dim(
   inp: Union[DataT, ArrayT]
)
```


----


### size
```python
.size(
   inp: Union[DataT, ArrayT], axis: AxisT = None
)
```


----


### flatten
```python
.flatten(
   inp: ArrayT
)
```


----


### reshape
```python
.reshape(
   inp: ArrayT, shape
)
```


----


### squeeze
```python
.squeeze(
   inp: ArrayT, axis: AxisT = None
)
```


----


### expand_dims
```python
.expand_dims(
   inp: ArrayT, axis
)
```


----


### pad
```python
.pad(
   inp: Union[DataT, ArrayT], padding, mode
)
```


----


### transpose
```python
.transpose(
   inp: Union[DataT, ArrayT], axes = None
)
```


----


### fill
```python
.fill(
   inp: ArrayT, value: DataT
)
```


----


### absolute
```python
.absolute(
   inp: Union[DataT, ArrayT], *, out: Optional[ArrayT] = None
)
```


----


### negative
```python
.negative(
   inp: Union[DataT, ArrayT], *, out: Optional[ArrayT] = None
)
```


----


### around
```python
.around(
   inp: Union[DataT, ArrayT], *, out: Optional[ArrayT] = None
)
```


----


### floor
```python
.floor(
   inp: Union[DataT, ArrayT], *, out: Optional[ArrayT] = None
)
```


----


### ceil
```python
.ceil(
   inp: Union[DataT, ArrayT], *, out: Optional[ArrayT] = None
)
```


----


### sqrt
```python
.sqrt(
   inp: Union[DataT, ArrayT], *, out: Optional[ArrayT] = None
)
```


----


### square
```python
.square(
   inp: Union[DataT, ArrayT], *, out: Optional[ArrayT] = None
)
```


----


### clip
```python
.clip(
   inp: Union[DataT, ArrayT], min_value: DataT, max_value: DataT, *,
   out: Optional[ArrayT] = None
)
```


----


### exp
```python
.exp(
   inp: Union[DataT, ArrayT], *, out: Optional[ArrayT] = None
)
```


----


### tanh
```python
.tanh(
   inp: Union[DataT, ArrayT], *, out: Optional[ArrayT] = None
)
```


----


### sum
```python
.sum(
   inp: ArrayT, axis: AxisT = None, keepdims = False
)
```


----


### mean
```python
.mean(
   inp: ArrayT, axis: AxisT = None
)
```


----


### median
```python
.median(
   inp: ArrayT, axis: AxisT = None
)
```


----


### var
```python
.var(
   inp: ArrayT, axis: AxisT = None
)
```


----


### std
```python
.std(
   inp: ArrayT, axis: AxisT = None
)
```


----


### prod
```python
.prod(
   inp: ArrayT, axis: AxisT = None
)
```


----


### unique
```python
.unique(
   inp: ArrayT, return_counts = False
)
```


----


### argmax
```python
.argmax(
   inp: ArrayT, axis: AxisT = None
)
```


----


### argmin
```python
.argmin(
   inp: ArrayT, axis: AxisT = None
)
```


----


### amax
```python
.amax(
   inp: ArrayT, axis: AxisT = None
)
```


----


### maximum
```python
.maximum(
   inp: ArrayT, axis: AxisT = None
)
```


----


### amin
```python
.amin(
   inp: ArrayT, axis: AxisT = None
)
```


----


### minimum
```python
.minimum(
   inp: ArrayT, axis: AxisT = None
)
```


----


### add
```python
.add(
   inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *,
   out: Optional[ArrayT] = None
)
```


----


### sub
```python
.sub(
   inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *,
   out: Optional[ArrayT] = None
)
```


----


### mul
```python
.mul(
   inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *,
   out: Optional[ArrayT] = None
)
```


----


### truediv
```python
.truediv(
   inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *,
   out: Optional[ArrayT] = None
)
```


----


### floordiv
```python
.floordiv(
   inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *,
   out: Optional[ArrayT] = None
)
```


----


### power
```python
.power(
   inp: Union[DataT, ArrayT], p: Union[DataT, ArrayT], *,
   out: Optional[ArrayT] = None
)
```


----


### lt
```python
.lt(
   inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *,
   out: Optional[ArrayT] = None
)
```


----


### le
```python
.le(
   inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *,
   out: Optional[ArrayT] = None
)
```


----


### gt
```python
.gt(
   inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *,
   out: Optional[ArrayT] = None
)
```


----


### ge
```python
.ge(
   inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *,
   out: Optional[ArrayT] = None
)
```


----


### eq
```python
.eq(
   inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *,
   out: Optional[ArrayT] = None
)
```


----


### ne
```python
.ne(
   inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *,
   out: Optional[ArrayT] = None
)
```


----


### dot
```python
.dot(
   inp1: ArrayT, inp2: ArrayT, *, out: Optional[ArrayT] = None
)
```

