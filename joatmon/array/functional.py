import math
from typing import (
    List,
    Optional,
    Tuple,
    Union
)

from joatmon.array.array import Array

__all__ = [
    'absolute', 'add', 'amax', 'maximum', 'amin', 'minimum', 'arange', 'argmax', 'argmin', 'around', 'array', 'asarray', 'astype', 'ceil',
    'clip', 'concatenate', 'copy', 'dim', 'dot', 'empty', 'eq', 'exp', 'expand_dims', 'eye', 'fill', 'flatten',
    'floor', 'floordiv', 'full', 'ge', 'getitem', 'gt', 'indices', 'le', 'linspace', 'lt', 'mean',
    'median', 'mul', 'ne', 'negative', 'ones', 'ones_like', 'pad', 'power', 'prod', 'put_along_axis', 'repeat',
    'reshape', 'setitem', 'size', 'split', 'sqrt', 'square', 'squeeze', 'stack', 'std', 'sub', 'sum',
    'take_along_axis', 'tanh', 'tolist', 'transpose', 'truediv', 'unique', 'var', 'where', 'zeros', 'zeros_like'
]

BooleanT = Union[bool]
IntegerT = Union[int]
FloatingT = Union[float]
NumberT = Union[IntegerT, FloatingT]

DataT = Union[BooleanT, IntegerT, FloatingT]
ArrayT = Union[Tuple[DataT], List[DataT], Array, Tuple['ArrayT'], List['ArrayT']]
TypeT = Union[str]
ShapeT = Union[Tuple[int], List[int]]
IndexT = Union[int, slice, Tuple[Union[int, slice], ...]]
AxisT = Optional[Union[int, Tuple[int], List[int]]]


class Broadcast:
    def __init__(self, inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT]):
        # both parameters are float, int, bool
        if dim(inp1) == 0 and dim(inp2) == 0:
            pass

        # first parameter is NDArray, second parameter is float, int, bool
        if dim(inp1) != 0 and dim(inp2) == 0:
            inp2 = reshape(repeat(inp2, prod(size(inp1))), size(inp1))

        # both parameters are NDArray
        if dim(inp1) != 0 and dim(inp2) != 0 and dim(inp1) != dim(inp2):
            # print('reshaping inp2 array')
            new_shape = [1] * (len(size(inp1)) - len(size(inp2))) + list(size(inp2))
            inp2 = reshape(inp2, new_shape)

        # first dimension sizes are the same, it can be broadcast together
        if dim(inp1) != 0 and dim(inp2) != 0 and size(inp1, 0) == size(inp2, 0):
            pass

        # first dimension sizes are not the same and first dimension size of the second input is not 1
        # print(inp1.shape, inp2.shape)
        if dim(inp1) != 0 and dim(inp2) != 0 and size(inp1, 0) != size(inp2, 0) and size(inp2, 0) != 1:
            raise ValueError('cannot broadcast')

        # first dimension sizes are not the same but the first dimension size of the second input is 1, it can be broadcast
        if dim(inp1) != 0 and dim(inp2) != 0 and size(inp1, 0) != size(inp2, 0) and size(inp2, 0) == 1:
            # print('repeating and reshaping inp2 array')
            inp2 = reshape(repeat(inp2, size(inp1, 0)), (size(inp1, 0),) + size(inp2)[1:])

        self.inp1 = inp1
        self.inp2 = inp2

        if dim(self.inp1) != 0 and dim(self.inp2) != 0:
            self.zip = zip(self.inp1, self.inp2)
        else:
            raise ValueError('an error has happened')

    def __iter__(self):
        return self.zip

    def __next__(self):
        return next(self.zip)

    def reset(self):
        if dim(self.inp1) != 0 and dim(self.inp2) != 0:
            self.zip = zip(self.inp1, self.inp2)
        else:
            raise ValueError('an error has happened')


def array(arr: ArrayT) -> ArrayT:
    return Array(arr)


def asarray(arr: ArrayT) -> ArrayT:
    return Array(arr)


def arange(start: float, stop: float, step: float) -> Array:
    return linspace(start, stop, int((stop - start) // step))


def linspace(start: float, stop: float, steps: int) -> Array:
    inc = (stop - start) / steps
    return Array([start + inc * idx for idx in range(steps)])


def eye(rows: int, columns: int) -> Array:
    # need to check which one is smaller
    ret = zeros((rows, columns))
    for row_col in range(min(rows, columns)):
        ret[row_col][row_col] = 1
    return ret


def empty(shape) -> Array:
    if len(shape) == 0:
        raise ValueError('array shape has to be at least 1-dimensional')

    ret = []
    for _ in range(shape[0]):
        ret.append(tolist(zeros(shape[1:])))
    return Array(ret)


def full(shape) -> Array:
    if len(shape) == 0:
        raise ValueError('array shape has to be at least 1-dimensional')

    ret = []
    for _ in range(shape[0]):
        ret.append(tolist(ones(shape[1:])))
    return Array(ret)


def zeros(shape) -> Array:
    if len(shape) == 0:
        raise ValueError('array shape has to be at least 1-dimensional')

    ret = []
    for _ in range(shape[0]):
        if len(shape[1:]) == 0:
            ret.append(float(0))
        else:
            ret.append(tolist(zeros(shape[1:])))
    return Array(ret)


def ones(shape) -> Array:
    if len(shape) == 0:
        raise ValueError('array shape has to be at least 1-dimensional')

    ret = []
    for _ in range(shape[0]):
        ret.append(tolist(ones(shape[1:])))
    return Array(ret)


def ones_like(inp: ArrayT) -> Array:
    return ones(size(inp))


def zeros_like(inp: ArrayT) -> Array:
    return zeros(size(inp))


def concatenate(inputs, axis: AxisT = None) -> ArrayT:
    ...


def stack(inputs, axis: AxisT = None) -> ArrayT:
    ...


def astype(inp: Union[DataT, ArrayT], dtype: TypeT) -> Union[DataT, ArrayT]:
    if isinstance(inp, (bool, int, float)):
        if dtype in ('bool', 'boolean'):
            return bool(inp)
        if dtype in ('int', 'integer', 'int8', 'int16', 'int32', 'int64'):
            return int(inp)
        if dtype in ('float', 'floating', 'float8', 'float16', 'float32', 'float64'):
            return float(inp)
    elif isinstance(inp, (tuple, list, Array)):
        ret = []
        for data in inp:
            ret.append(astype(data, dtype))
        return type(inp)(ret)
    else:
        raise ValueError('value type is not recognized')


def copy(inp: Union[DataT, ArrayT]) -> Union[DataT, ArrayT]:
    if isinstance(inp, (bool, int, float)):
        return type(inp)(inp)
    elif isinstance(inp, (tuple, list, Array)):
        return type(inp)(tolist(inp))
    else:
        raise ValueError(f'cannot copy object of type {type(inp)}')


def repeat(inp: Union[DataT, ArrayT], count: DataT, axis=0) -> ArrayT:
    # need to use axis
    # need to be able to return tuple or list

    ret = []
    for _ in range(count):
        if isinstance(inp, (bool, int, float)):
            ret.append(inp)
        elif isinstance(inp, (tuple, list, Array)):
            ret.append(tolist(inp))
    return Array(ret)


def split(inp: ArrayT, chunks, axis=0) -> ArrayT:
    # need to use axis
    # need to return tuple

    length = len(inp)
    chunk_length = length // chunks
    if chunk_length * chunks != length:
        raise ValueError(f'array with length {length} cannot be divided into {chunks} chunks')

    ret = []
    for chunk in range(chunks):
        ret.append(inp[chunk * chunk_length: (chunk + 1) * chunk_length])
    return type(inp)(ret)


def tolist(inp: ArrayT) -> list:
    ret = []
    for data in inp:
        if isinstance(data, (bool, int, float)):
            ret.append(data)
        elif isinstance(data, (tuple, list, Array)):
            ret.append(tolist(data))
        else:
            raise ValueError(f'{type(data)} could not be converted')
    return ret


def getitem(inp: ArrayT, idx: IndexT):
    if isinstance(inp, (tuple, list)):
        if isinstance(idx, int):
            return copy(inp[idx])
        elif isinstance(idx, slice):
            ret = []
            for data in inp[idx]:
                ret.append(data)
            return type(inp)(ret)
        elif isinstance(idx, (tuple, list)) and len(idx) == 1:
            ret = inp[idx[0]]
            if isinstance(ret, (bool, int, float)):
                return ret
            elif isinstance(ret, (tuple, list, Array)):
                return type(inp)(ret)
        elif isinstance(idx, (tuple, list)):
            ret = []
            if isinstance(idx[0], int):
                return getitem(inp[idx[0]], idx[1:])
            elif isinstance(idx[0], slice):
                for data in inp[idx[0]]:
                    ret.append(getitem(data, idx[1:]))
            return type(inp)(ret)
    elif isinstance(inp, Array):
        ret = getitem(tolist(inp), idx)
        if isinstance(ret, (bool, int, float)):
            return ret
        return Array(ret)
    else:
        raise ValueError(f'object type {type(inp)} is not recognized')


def take_along_axis(inp: Union[DataT, ArrayT], indexes, axis) -> ArrayT:
    ...


def setitem(inp: ArrayT, idx: IndexT, value: Union[DataT, ArrayT]):
    # need to repeat inp[idx].shape times
    if isinstance(value, (bool, int, float)):
        inp_shape = size(inp)
        inp_size = prod(inp_shape)
        new_value = reshape(repeat(value, inp_size), inp_shape)
    elif isinstance(value, (tuple, list)):
        if size(inp) != size(value):
            raise ValueError(f'{size(inp)} and {size(value)} does not match')
        new_value = copy(value)
    elif isinstance(value, Array):
        if size(inp) != size(value):
            raise ValueError(f'{size(inp)} and {size(value)} does not match')
        new_value = copy(value)
    else:
        raise ValueError(f'{type(value)} is not recognized as value type')

    if isinstance(inp, (tuple, list)):
        if isinstance(idx, int):
            inp[idx] = new_value[idx]
        elif isinstance(idx, slice):
            for _idx in range(idx.start, idx.stop, idx.step):
                inp[_idx] = new_value[_idx]
        elif isinstance(idx, (tuple, list)) and len(idx) == 1:
            inp[idx[0]] = new_value[idx[0]]
        elif isinstance(idx, (tuple, list)):
            if isinstance(idx[0], int):
                setitem(inp[idx[0]], idx[1:], new_value[idx[0]])
            elif isinstance(idx[0], slice):
                for _idx in range(idx[0].start, idx[0].stop, idx[0].step):
                    setitem(inp[_idx], idx[1:], new_value[_idx])
            else:
                raise ValueError(f'{type(idx[0])} type is not recognized as index')
        else:
            raise ValueError(f'{type(idx)} type is not recognized as index')
        # does not work when not returning
    elif isinstance(inp, Array):
        list_inp = tolist(inp)
        setitem(list_inp, idx, new_value)
        return Array(list_inp)
    else:
        raise ValueError(f'object type {type(inp)} is not recognized')


def put_along_axis(inp: Union[DataT, ArrayT], indexes, values, axis) -> ArrayT:
    ...


def where(condition) -> ArrayT:
    ...


def indices(dimensions) -> ArrayT:
    ...


def dim(inp: Union[DataT, ArrayT]) -> int:
    if isinstance(inp, (bool, int, float)):
        return 0
    return dim(inp[0]) + 1


def size(inp: Union[DataT, ArrayT], axis: AxisT = None):
    if isinstance(inp, (bool, int, float)):
        return ()

    if axis is None or axis < 0:
        return tuple([len(inp)] + list(size(inp[0])))
    return tuple((size(inp[0], axis=axis - 1)))


def flatten(inp: ArrayT) -> ArrayT:
    data_type = type(inp)

    ret = []
    for data in inp:
        if dim(data) == 0:
            ret.append(data)
        elif dim(data) == 1:
            ret += tolist(data)
        else:
            ret += tolist(flatten(data))
    return data_type(ret)


def reshape(inp: ArrayT, shape) -> ArrayT:
    flat = flatten(inp)

    subdims = shape[1:]
    subsize = prod(Array(subdims))
    if shape[0] * subsize != len(flat):
        raise ValueError('size does not match or invalid')
    if not subdims:
        return flat
    return Array([reshape(flat[i: i + subsize], subdims) for i in range(0, len(flat), subsize)])


def squeeze(inp: ArrayT, axis: AxisT = None) -> ArrayT:
    if axis is None:
        if any([x == 1 for x in size(inp)]):
            new_shape = tuple(filter(lambda x: x != 1, size(inp)))
        else:
            new_shape = size(inp)
    else:
        axis_size = size(inp, axis)
        if axis_size != 1:
            raise ValueError('cannot select an axis to squeeze out which has size not equal to one')
        new_shape = list(size(inp))
        new_shape.pop(axis)

    return reshape(inp, new_shape)


def expand_dims(inp: ArrayT, axis) -> ArrayT:
    shape = list(size(inp))
    shape.insert(axis, 1)

    return reshape(inp, shape)


def pad(inp: Union[DataT, ArrayT], padding, mode) -> ArrayT:
    ...


def transpose(inp: Union[DataT, ArrayT], axes=None) -> ArrayT:
    def multi_to_one(idx, s):
        r = 0
        for i, _idx in enumerate(idx):
            r += (prod(s[i + 1:]) * _idx)
        return r

    def one_to_multi(idx, s):
        r = []
        for i in range(len(s)):
            div, mod = divmod(idx, prod(s[i + 1:]))
            r.append(div)
            idx = mod
        return r

    if axes is None:
        if dim(inp) != 2:
            raise ValueError(f'axes has to be used on {dim(inp)}-dim array')
        axes = (1, 0)

    shape = size(inp)
    new_shape = []
    for axis in axes:
        new_shape.append(shape[axis])
    new_shape = tuple(new_shape)

    new_flatten = flatten(zeros(new_shape))
    old_flatten = flatten(inp)

    for flat_idx in range(len(old_flatten)):
        old_multi = one_to_multi(flat_idx, shape)
        new_multi = []
        for axis in axes:
            new_multi.append(old_multi[axis])

        old_flat_idx = multi_to_one(old_multi, shape)
        new_flat_idx = multi_to_one(new_multi, new_shape)

        new_flatten = setitem(new_flatten, new_flat_idx, getitem(old_flatten, old_flat_idx))

    return reshape(new_flatten, new_shape)


def fill(inp: ArrayT, value: DataT) -> ArrayT:
    shape = size(inp)
    flat_array = flatten(inp)
    for idx in range(len(flat_array)):
        setitem(flat_array, idx, value)
    return reshape(flat_array, shape)


def absolute(inp: Union[DataT, ArrayT], *, out: Optional[ArrayT] = None) -> Union[DataT, ArrayT]:
    flat_array = flatten(inp)
    for idx in range(len(flat_array)):
        setitem(flat_array, idx, abs(getitem(flat_array, idx)))
    return reshape(flat_array, size(inp))


def negative(inp: Union[DataT, ArrayT], *, out: Optional[ArrayT] = None) -> Union[DataT, ArrayT]:
    flat_array = flatten(inp)
    for idx in range(len(flat_array)):
        setitem(flat_array, idx, -getitem(flat_array, idx))
    return reshape(flat_array, size(inp))


def around(inp: Union[DataT, ArrayT], *, out: Optional[ArrayT] = None) -> Union[DataT, ArrayT]:
    flat_array = flatten(inp)
    for idx in range(len(flat_array)):
        setitem(flat_array, idx, round(getitem(flat_array, idx)))
    return reshape(flat_array, size(inp))


def floor(inp: Union[DataT, ArrayT], *, out: Optional[ArrayT] = None) -> Union[DataT, ArrayT]:
    flat_array = flatten(inp)
    for idx in range(len(flat_array)):
        setitem(flat_array, idx, math.floor(getitem(flat_array, idx)))
    return reshape(flat_array, size(inp))


def ceil(inp: Union[DataT, ArrayT], *, out: Optional[ArrayT] = None) -> Union[DataT, ArrayT]:
    flat_array = flatten(inp)
    for idx in range(len(flat_array)):
        setitem(flat_array, idx, math.ceil(getitem(flat_array, idx)))
    return reshape(flat_array, size(inp))


def sqrt(inp: Union[DataT, ArrayT], *, out: Optional[ArrayT] = None) -> Union[DataT, ArrayT]:
    flat_array = flatten(inp)
    for idx in range(len(flat_array)):
        setitem(flat_array, idx, math.sqrt(getitem(flat_array, idx)))
    return reshape(flat_array, size(inp))


def square(inp: Union[DataT, ArrayT], *, out: Optional[ArrayT] = None) -> Union[DataT, ArrayT]:
    flat_array = flatten(inp)
    for idx in range(len(flat_array)):
        setitem(flat_array, idx, getitem(flat_array, idx) ** 2)
    return reshape(flat_array, size(inp))


def clip(inp: Union[DataT, ArrayT], min_value: DataT, max_value: DataT, *, out: Optional[ArrayT] = None) -> Union[DataT, ArrayT]:
    flat_array = flatten(inp)
    for idx in range(len(flat_array)):
        value = getitem(flat_array, idx)
        if value < min_value:
            value = min_value
        if value > max_value:
            value = max_value
        setitem(flat_array, idx, value)
    return reshape(flat_array, size(inp))


def exp(inp: Union[DataT, ArrayT], *, out: Optional[ArrayT] = None) -> Union[DataT, ArrayT]:
    flat_array = flatten(inp)
    for idx in range(len(flat_array)):
        setitem(flat_array, idx, math.e ** getitem(flat_array, idx))
    return reshape(flat_array, size(inp))


def tanh(inp: Union[DataT, ArrayT], *, out: Optional[ArrayT] = None) -> Union[DataT, ArrayT]:
    flat_array = flatten(inp)
    for idx in range(len(flat_array)):
        setitem(flat_array, idx, math.tanh(getitem(flat_array, idx)))
    return reshape(flat_array, size(inp))


def sum(inp: ArrayT, axis: AxisT = None, keepdims=False) -> Union[DataT, ArrayT]:
    if axis is None:
        if dim(inp) == 0:
            return astype(copy(inp), 'float')

        ret = float(0)
        for idx in range(len(inp)):
            ret += sum(inp[idx], axis=axis)
        return ret

    # kahan sum
    shape = size(inp)
    s = zeros(shape[:axis] + shape[axis + 1:])
    c = zeros(size(s))
    for i in range(shape[axis]):
        # y = getitem(inp, (slice(None),) * axis + (i,)) - c  # inp[(slice(None),) * axis + (i,)] - c
        y = inp[(slice(None),) * axis + (i,)] - c
        t = s + y
        c = (t - s) - y
        s = copy(t)
    return s


def mean(inp: ArrayT, axis: AxisT = None) -> Union[DataT, ArrayT]:
    shape = size(inp)
    s = sum(inp, axis)
    n = shape[axis] if axis is not None else prod(shape)
    return s / n


def median(inp: ArrayT, axis: AxisT = None) -> Union[DataT, ArrayT]:
    ...


def var(inp: ArrayT, axis: AxisT = None) -> Union[DataT, ArrayT]:
    if axis is not None:
        m_shape = list(size(inp))
        m_shape[axis] = 1
        m = reshape(mean(inp, axis), m_shape)
        a = absolute(inp - m)
        return mean(a ** 2, axis=axis)

    return mean(absolute(inp - mean(inp)) ** 2)


def std(inp: ArrayT, axis: AxisT = None) -> Union[DataT, ArrayT]:
    return sqrt(var(inp, axis=axis))


def prod(inp: ArrayT, axis: AxisT = None) -> int:
    # could be multi dimensional
    p = 1
    if isinstance(inp, (tuple, list)):
        for data in inp:
            p *= data
    elif isinstance(inp, Array):
        return prod(inp.data)
    return p


def unique(inp: ArrayT, return_counts=False) -> ArrayT:
    ...


def argmax(inp: ArrayT, axis: AxisT = None) -> Union[DataT, ArrayT]:
    ...


def argmin(inp: ArrayT, axis: AxisT = None) -> Union[DataT, ArrayT]:
    ...


def amax(inp: ArrayT, axis: AxisT = None) -> Union[DataT, ArrayT]:
    ...


def maximum(inp: ArrayT, axis: AxisT = None) -> Union[DataT, ArrayT]:
    ...


def amin(inp: ArrayT, axis: AxisT = None) -> Union[DataT, ArrayT]:
    ...


def minimum(inp: ArrayT, axis: AxisT = None) -> Union[DataT, ArrayT]:
    ...


def add(inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *, out: Optional[ArrayT] = None) -> Union[DataT, ArrayT]:
    if dim(inp1) == 0 and dim(inp2) == 0:
        return inp1 + inp2

    ret = out or zeros(size(inp1))
    broadcast = Broadcast(inp1, inp2)

    for idx, (_inp1, _inp2) in enumerate(broadcast):
        ret[idx] = add(_inp1, _inp2, out=ret[idx])
    return ret


def sub(inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *, out: Optional[ArrayT] = None) -> Union[DataT, ArrayT]:
    if dim(inp1) == 0 and dim(inp2) == 0:
        return inp1 - inp2

    ret = out or zeros(size(inp1))
    broadcast = Broadcast(inp1, inp2)

    for idx, (_inp1, _inp2) in enumerate(broadcast):
        ret[idx] = sub(_inp1, _inp2, out=ret[idx])
    return ret


def mul(inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *, out: Optional[ArrayT] = None) -> Union[DataT, ArrayT]:
    if dim(inp1) == 0 and dim(inp2) == 0:
        return inp1 * inp2

    ret = out or zeros(size(inp1))
    broadcast = Broadcast(inp1, inp2)

    for idx, (_inp1, _inp2) in enumerate(broadcast):
        ret[idx] = mul(_inp1, _inp2, out=ret[idx])
    return ret


def truediv(inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *, out: Optional[ArrayT] = None) -> Union[DataT, ArrayT]:
    if dim(inp1) == 0 and dim(inp2) == 0:
        return inp1 / inp2

    ret = out or zeros(size(inp1))
    broadcast = Broadcast(inp1, inp2)

    for idx, (_inp1, _inp2) in enumerate(broadcast):
        ret[idx] = truediv(_inp1, _inp2, out=ret[idx])
    return ret


def floordiv(inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *, out: Optional[ArrayT] = None) -> Union[DataT, ArrayT]:
    if dim(inp1) == 0 and dim(inp2) == 0:
        return inp1 // inp2

    ret = out or zeros(size(inp1))
    broadcast = Broadcast(inp1, inp2)

    for idx, (_inp1, _inp2) in enumerate(broadcast):
        ret[idx] = floordiv(_inp1, _inp2, out=ret[idx])
    return ret


def power(inp: Union[DataT, ArrayT], p: Union[DataT, ArrayT], *, out: Optional[ArrayT] = None) -> Union[DataT, ArrayT]:
    if dim(inp) == 0:
        return inp ** p

    ret = out or zeros(size(inp))
    for idx, _inp in enumerate(inp):
        ret[idx] = power(_inp, p, out=ret[idx])
    return ret


def lt(inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *, out: Optional[ArrayT] = None) -> Union[DataT, ArrayT]:
    if dim(inp1) == 0 and dim(inp2) == 0:
        return inp1 < inp2

    ret = out or zeros(size(inp1))
    broadcast = Broadcast(inp1, inp2)

    for idx, (_inp1, _inp2) in enumerate(broadcast):
        ret[idx] = lt(_inp1, _inp2, out=ret[idx])
    return ret


def le(inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *, out: Optional[ArrayT] = None) -> Union[DataT, ArrayT]:
    if dim(inp1) == 0 and dim(inp2) == 0:
        return inp1 <= inp2

    ret = out or zeros(size(inp1))
    broadcast = Broadcast(inp1, inp2)

    for idx, (_inp1, _inp2) in enumerate(broadcast):
        ret[idx] = le(_inp1, _inp2, out=ret[idx])
    return ret


def gt(inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *, out: Optional[ArrayT] = None) -> Union[DataT, ArrayT]:
    if dim(inp1) == 0 and dim(inp2) == 0:
        return inp1 > inp2

    ret = out or zeros(size(inp1))
    broadcast = Broadcast(inp1, inp2)

    for idx, (_inp1, _inp2) in enumerate(broadcast):
        ret[idx] = gt(_inp1, _inp2, out=ret[idx])
    return ret


def ge(inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *, out: Optional[ArrayT] = None) -> Union[DataT, ArrayT]:
    if dim(inp1) == 0 and dim(inp2) == 0:
        return inp1 >= inp2

    ret = out or zeros(size(inp1))
    broadcast = Broadcast(inp1, inp2)

    for idx, (_inp1, _inp2) in enumerate(broadcast):
        ret[idx] = ge(_inp1, _inp2, out=ret[idx])
    return ret


def eq(inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *, out: Optional[ArrayT] = None) -> Union[DataT, ArrayT]:
    if dim(inp1) == 0 and dim(inp2) == 0:
        return inp1 == inp2

    ret = out or zeros(size(inp1))
    broadcast = Broadcast(inp1, inp2)

    for idx, (_inp1, _inp2) in enumerate(broadcast):
        ret[idx] = eq(_inp1, _inp2, out=ret[idx])
    return ret


def ne(inp1: Union[DataT, ArrayT], inp2: Union[DataT, ArrayT], *, out: Optional[ArrayT] = None) -> Union[DataT, ArrayT]:
    if dim(inp1) == 0 and dim(inp2) == 0:
        return inp1 != inp2

    ret = out or zeros(size(inp1))
    broadcast = Broadcast(inp1, inp2)

    for idx, (_inp1, _inp2) in enumerate(broadcast):
        ret[idx] = ne(_inp1, _inp2, out=ret[idx])
    return ret


def dot(inp1: ArrayT, inp2: ArrayT, *, out: Optional[ArrayT] = None) -> ArrayT:
    ...
