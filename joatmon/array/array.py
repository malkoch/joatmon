import itertools

from joatmon.array import functional


class Array:
    """
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
    """

    def __init__(self, data=None, dtype='float32'):
        if data is None:
            raise ValueError(f'data cannot be {data}')

        self._data = functional.flatten(data)
        self._dtype = dtype
        self._shape = functional.size(data)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f'Array(dtype={self.dtype}, shape={self.shape}, data={functional.reshape(self._data, self._shape)})'

    def __bool__(self):
        return True

    def __copy__(self):
        ...

    def __deepcopy__(self, memodict=None):
        ...

    def __iter__(self):
        for data in self._data:
            yield data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
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
        ...

    def astype(self, dtype, copy=True):
        ...

    # noinspection PyPep8Naming
    @property
    def T(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return self

    @property
    def ndim(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return len(self.shape)

    @property
    def shape(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return self._shape

    @property
    def dtype(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return self._dtype
