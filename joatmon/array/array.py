def _check_input(data):
    def _helper(_data, _length=None, _dtype=None):
        data_length = None
        data_type = type(_data)

        if isinstance(_data, (tuple, list)):
            data_length = len(_data)
            if len(_data) == 0:
                raise ValueError('empty list cannot be used')

            if _length is not None and len(_data) != _length:
                raise ValueError(f'shape mismatch')

            _check_input(_data)
        elif not isinstance(_data, (bool, int, float)):
            raise ValueError('data has to be one of the following types: (bool, int, float)')

        if _dtype is not None and type(_data) != _dtype:
            raise ValueError(f'data type mismatch between {type(_data)} and {_dtype}')

        return data_length, data_type

    if not isinstance(data, (tuple, list)):
        raise ValueError('input data has to be either tuple or list type')

    length = None
    dtype = None
    for _data in data:
        length, dtype = _helper(_data, length, dtype)


def _normalize_input(data):
    return data


class Array:
    def __init__(self, data=None):
        _check_input(data)
        _normalize_input(data)

        self._data = data
        self._device = 'cpu'
        self._dtype = 'float32'

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f'Array(data={self.data}, dtype={self.dtype}, shape={self.shape})'

    def __bool__(self):
        return True

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memodict=None):
        return self.copy()

    def __iter__(self):
        for data in self.data:
            yield data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.getitem(idx)

    def __setitem__(self, key, value):
        self.setitem(key, value)

    def __abs__(self):
        return self.abs()

    def __neg__(self):
        return self.negative()

    def __round__(self, n=None):
        pass

    def __floor__(self):
        pass

    def __ceil__(self):
        pass

    def __add__(self, other):
        return self.add(other)

    def __iadd__(self, other):
        return self.add(other, out=self)

    def __sub__(self, other):
        return self.sub(other)

    def __isub__(self, other):
        return self.sub(other, out=self)

    def __mul__(self, other):
        return self.mul(other)

    def __imul__(self, other):
        return self.mul(other, out=self)

    def __truediv__(self, other):
        return self.truediv(other)

    def __itruediv__(self, other):
        return self.truediv(other, out=self)

    def __floordiv__(self, other):
        return self.floordiv(other)

    def __ifloordiv__(self, other):
        return self.floordiv(other, out=self)

    def __pow__(self, power, modulo=None):
        return self.power(power)

    def __ipow__(self, other):
        return self.power(other, out=self)

    def __lt__(self, other):
        return self.lt(other)

    def __le__(self, other):
        return self.le(other)

    def __gt__(self, other):
        return self.gt(other)

    def __ge__(self, other):
        return self.ge(other)

    def __eq__(self, other):
        return self.eq(other)

    def __ne__(self, other):
        return self.ne(other)

    def astype(self, dtype):
        ...

    def copy(self):
        ...

    def repeat(self):
        ...

    def split(self):
        ...

    def tolist(self):
        ...

    def getitem(self, idx):
        ...

    def take_along_axis(self):
        ...

    def setitem(self, idx, value):
        ...

    def put_along_axis(self):
        ...

    def where(self):
        ...

    def indices(self):
        ...

    def dim(self):
        ...

    def size(self, dim=None):
        ...

    def flatten(self):
        ...

    def reshape(self, size):
        ...

    def squeeze(self):
        ...

    def expand_dims(self):
        ...

    def pad(self):
        ...

    def transpose(self):
        ...

    def fill(self, value):
        ...

    def abs(self):
        ...

    def negative(self, *, out=None):
        ...

    def round(self, *, out=None):
        ...

    def floor(self, *, out=None):
        ...

    def ceil(self, *, out=None):
        ...

    def sqrt(self, *, out=None):
        ...

    def square(self, *, out=None):
        ...

    def clip(self, min_value, max_value, *, out=None):
        ...

    def exp(self, *, out=None):
        ...

    def tanh(self, *, out=None):
        ...

    def sum(self):
        ...

    def mean(self):
        ...

    def median(self):
        ...

    def var(self):
        ...

    def std(self):
        ...

    def prod(self):
        ...

    def unique(self):
        ...

    def argmax(self):
        ...

    def argmin(self):
        ...

    def amax(self):
        ...

    def amin(self):
        ...

    def add(self, other, *, out=None):
        ...

    def sub(self, other, *, out=None):
        ...

    def mul(self, other, *, out=None):
        ...

    def truediv(self, other, *, out=None):
        ...

    def floordiv(self, other, *, out=None):
        ...

    def power(self, p, *, out=None):
        ...

    def lt(self, other):
        ...

    def le(self, other):
        ...

    def gt(self, other):
        ...

    def ge(self, other):
        ...

    def eq(self, other):
        ...

    def ne(self, other):
        ...

    @property
    def device(self):
        return self._device

    @property
    def data(self):
        return self._data

    # noinspection PyPep8Naming
    @property
    def T(self):
        return self.transpose()

    @property
    def ndim(self):
        return self.dim()

    @property
    def shape(self):
        return self.size()

    @property
    def dtype(self):
        return self._dtype
