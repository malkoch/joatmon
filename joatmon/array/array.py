import itertools

from joatmon.array import functional


class Array:
    """
    Array class for handling multi-dimensional data.

    This class provides a way to handle multi-dimensional data in a flattened format,
    while still providing access to the data in its original shape.

    Attributes:
        _data (list): The flattened data.
        _dtype (str): The data type of the elements in the array.
        _shape (tuple): The shape of the original data.

    Args:
        data (list): The original multi-dimensional data.
        dtype (str): The data type of the elements in the array.
    """

    def __init__(self, data=None, dtype='float32'):
        """
        Initialize the Array.

        Args:
            data (list): The original multi-dimensional data.
            dtype (str): The data type of the elements in the array.
        """
        if data is None:
            raise ValueError(f'data cannot be {data}')

        self._data = functional.flatten(data)
        self._dtype = dtype
        self._shape = functional.size(data)

    def __repr__(self):
        """
        Return a string representation of the Array.
        """
        return str(self)

    def __str__(self):
        """
        Return a string representation of the Array.
        """
        return f'Array(dtype={self.dtype}, shape={self.shape}, data={functional.reshape(self._data, self._shape)})'

    def __bool__(self):
        """
        Return True for non-empty Array.
        """
        return True

    def __copy__(self):
        """
        Return a shallow copy of the Array.
        """

    def __deepcopy__(self, memodict=None):
        """
        Return a deep copy of the Array.
        """

    def __iter__(self):
        """
        Return an iterator over the elements of the Array.
        """
        for data in self._data:
            yield data

    def __len__(self):
        """
        Return the number of elements in the Array.
        """
        return len(self._data)

    def __getitem__(self, index):
        """
        Return the element(s) at the specified index.

        Args:
            index (int or tuple): The index of the element(s) to be returned.
        """
        indices_list = []
        if isinstance(index, int):
            index = [index]  # need to drop the first dimension
        index = list(index)
        if len(index) < self.ndim:
            index += [slice(0, x, 1) for x in self.shape[len(index):]]

        for index_value in index:
            if isinstance(index_value, slice):
                indices_list.append(list(range(index_value.start or 0, index_value.stop, index_value.step or 1)))
            elif isinstance(index_value, int):
                indices_list.append([index_value])

        output_shape = [len(x) for x in indices_list]

        ret = []
        for idx in itertools.product(*indices_list):
            ret.append(self._data[functional.ravel_index(idx, self._shape)])
        return Array(functional.reshape(ret, output_shape))

    def __setitem__(self, index, value):
        """
        Set the element(s) at the specified index to the given value.

        Args:
            index (int or tuple): The index of the element(s) to be set.
            value (various types): The value to be set.
        """

    def astype(self, dtype, copy=True):
        """
        Convert the Array to a specified data type.

        Args:
            dtype (str): The data type to convert to.
            copy (bool): Whether to return a new Array or modify the existing one.
        """

    # noinspection PyPep8Naming
    @property
    def T(self):
        """
        Return the transpose of the Array.
        """
        return self

    @property
    def ndim(self):
        """
        Return the number of dimensions of the Array.
        """
        return len(self.shape)

    @property
    def shape(self):
        """
        Return the shape of the Array.
        """
        return self._shape

    @property
    def dtype(self):
        """
        Return the data type of the elements in the Array.
        """
        return self._dtype
