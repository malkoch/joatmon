import operator
import warnings
import weakref
from collections import (
    defaultdict,
    OrderedDict
)
from functools import wraps
from itertools import islice
from typing import (
    Any,
    Callable,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union
)

import six

from joatmon.nn.utility import (
    _forward_unimplemented,
    EPOCH_DEPRECATION_WARNING,
    indent,
    legacy_get_string,
    required,
    typename,
)

__all__ = [
    'Tensor',
    'Module',
    'Parameter',
    'Optimizer',
    'LRScheduler',
    'Loss',
    'ModuleAttributeException',
    'Sequential',
]

warnings.filterwarnings(
    'once',
    'The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad '
    "attribute won't be populated during autograd.backward(). If you indeed want the gradient "
    'for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the '
    'non-leaf Tensor by mistake, make sure you access the leaf Tensor instead.',
    UserWarning,
)


class ModuleAttributeException(AttributeError):
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


class RemovableHandle(object):
    """A handle which provides the capability to remove a hook."""

    id: int
    next_id: int = 0

    def __init__(self, hooks_dict: Any) -> None:
        self.hooks_dict_ref = weakref.ref(hooks_dict)
        self.id = RemovableHandle.next_id
        RemovableHandle.next_id += 1

    def remove(self) -> None:
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        hooks_dict = self.hooks_dict_ref()
        if hooks_dict is not None and self.id in hooks_dict:
            del hooks_dict[self.id]

    def __getstate__(self):
        return self.hooks_dict_ref(), self.id

    def __setstate__(self, state) -> None:
        if state[0] is None:
            # create a dead reference
            self.hooks_dict_ref = weakref.ref(OrderedDict())
        else:
            self.hooks_dict_ref = weakref.ref(state[0])
        self.id = state[1]
        RemovableHandle.next_id = max(RemovableHandle.next_id, self.id + 1)

    def __enter__(self) -> 'RemovableHandle':
        return self

    def __exit__(self, type: Any, value: Any, tb: Any) -> None:
        self.remove()


class Tensor:
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

    def __init__(self, data=None, requires_grad=None):
        self._data = data
        self._requires_grad = requires_grad
        self._grad_fn = None
        self._grad = None
        self._name = None
        self._backward_hooks = None
        self._device = 'cpu'

    def __repr__(self):
        return str(self)

    def __str__(self):
        name = f'name={self._name}' if self._name is not None else ''
        data = f'data={self._data}'
        require_grad = f'requires_grad={self._requires_grad}' if self._requires_grad is not None else ''
        grad_func = f'grad_fn=<{self._grad_fn.__name__}>' if self._grad_fn is not None else ''
        device = f'device={self.device}'

        str_list = [name, device, data, require_grad, grad_func]
        str_list = list(filter(lambda parameter: parameter != '', str_list))
        string = ', '.join(str_list)

        return f'Tensor({string})'

    def __hash__(self):
        return id(self)

    def __getitem__(self, item):
        data = self.data[item]
        tensor = Tensor.from_array(data, requires_grad=self.requires_grad)
        tensor._grad_fn = self._grad_fn
        if self.device == 'gpu':
            tensor = tensor.gpu()
        return tensor

    def __setitem__(self, key, value):
        # self._grad_fn = value._grad_fn
        # self._requires_grad = value.requires_grad
        self._data[key] = value.data

    def __ge__(self, other):
        # return self.greater_or_equal(other)
        if isinstance(other, Tensor):
            return Tensor(data=self.data >= other.data)
        else:
            return Tensor(data=self.data >= other)

    def __gt__(self, other):
        # return self.greater(other)
        if isinstance(other, Tensor):
            return Tensor(data=self.data > other.data)
        else:
            return Tensor(data=self.data > other)

    def __le__(self, other):
        # return self.lesser_or_equal(other)
        if isinstance(other, Tensor):
            return Tensor(data=self.data <= other.data)
        else:
            return Tensor(data=self.data <= other)

    def __lt__(self, other):
        # return self.lesser(other)
        if isinstance(other, Tensor):
            return Tensor(data=self.data < other.data)
        else:
            return Tensor(data=self.data < other)

    def __eq__(self, other):
        # return self.equal(other)
        if isinstance(other, Tensor):
            return Tensor(data=self.data == other.data)
        else:
            return Tensor(data=self.data == other)

    def __ne__(self, other):
        # return self.not_equal(other)
        if isinstance(other, Tensor):
            return Tensor(data=self.data != other.data)
        else:
            return Tensor(data=self.data != other)

    def __add__(self, other) -> 'Tensor':
        return self.add(other)

    def __radd__(self, other):
        return self.add(other)

    def __sub__(self, other) -> 'Tensor':
        return self.sub(other)

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other) -> 'Tensor':
        return self.mul(other)

    def __rmul__(self, other):
        return self.mul(other)

    def __truediv__(self, other) -> 'Tensor':
        return self.div(other)

    def __pow__(self, power, modulo=None) -> 'Tensor':
        return self.power(power)

    def __abs__(self) -> 'Tensor':
        return self.absolute()

    def __neg__(self):
        return self.negative()

    def chunk(self, chunks, dim=0):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...

    def view(self, size) -> 'Tensor':
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...

    def index_select(self, dim, index) -> 'Tensor':
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...

    def zero(self) -> 'Tensor':
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...

    def one(self) -> 'Tensor':
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...

    def fill(self, value) -> 'Tensor':
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...

    def squeeze(self, axis=None) -> 'Tensor':
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...

    def expand_dim(self, axis=None) -> 'Tensor':
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...

    def transpose(self, axes) -> 'Tensor':
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...

    def absolute(self) -> 'Tensor':
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...

    def around(self) -> 'Tensor':
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...

    def floor(self) -> 'Tensor':
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...

    def ceil(self) -> 'Tensor':
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...

    def clip(self, min_val, max_val) -> 'Tensor':
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...

    def negative(self) -> 'Tensor':
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...

    def log(self) -> 'Tensor':
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...

    def summation(self) -> 'Tensor':
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...

    def mean(self) -> 'Tensor':
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...

    def std(self) -> 'Tensor':
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...

    def var(self) -> 'Tensor':
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...

    # def greater_or_equal(self, other) -> 'Tensor':
    #     """
    #     Remember the transaction.
    #
    #     Accepts a state, action, reward, next_state, terminal transaction.
    #
    #     # Arguments
    #         transaction (abstract): state, action, reward, next_state, terminal transaction.
    #     """
    #     ...
    #
    # def greater(self, other) -> 'Tensor':
    #     """
    #     Remember the transaction.
    #
    #     Accepts a state, action, reward, next_state, terminal transaction.
    #
    #     # Arguments
    #         transaction (abstract): state, action, reward, next_state, terminal transaction.
    #     """
    #     ...
    #
    # def lesser_or_equal(self, other) -> 'Tensor':
    #     """
    #     Remember the transaction.
    #
    #     Accepts a state, action, reward, next_state, terminal transaction.
    #
    #     # Arguments
    #         transaction (abstract): state, action, reward, next_state, terminal transaction.
    #     """
    #     ...
    #
    # def lesser(self, other) -> 'Tensor':
    #     """
    #     Remember the transaction.
    #
    #     Accepts a state, action, reward, next_state, terminal transaction.
    #
    #     # Arguments
    #         transaction (abstract): state, action, reward, next_state, terminal transaction.
    #     """
    #     ...
    #
    # def equal(self, other) -> 'Tensor':
    #     """
    #     Remember the transaction.
    #
    #     Accepts a state, action, reward, next_state, terminal transaction.
    #
    #     # Arguments
    #         transaction (abstract): state, action, reward, next_state, terminal transaction.
    #     """
    #     ...
    #
    # def not_equal(self, other) -> 'Tensor':
    #     """
    #     Remember the transaction.
    #
    #     Accepts a state, action, reward, next_state, terminal transaction.
    #
    #     # Arguments
    #         transaction (abstract): state, action, reward, next_state, terminal transaction.
    #     """
    #     ...

    def add(self, other) -> 'Tensor':
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...

    def sub(self, other) -> 'Tensor':
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...

    def mul(self, other) -> 'Tensor':
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...

    def div(self, other) -> 'Tensor':
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...

    def power(self, p) -> 'Tensor':
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...

    def clone(self) -> 'Tensor':
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...

    def detach(self, inplace=False) -> 'Tensor':
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...

    @staticmethod
    def from_array(data, requires_grad=False) -> 'Tensor':
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...

    def to_array(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...

    def half(self) -> 'Tensor':
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...

    def single(self) -> 'Tensor':
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...

    def double(self) -> 'Tensor':
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...

    def cpu(self) -> 'Tensor':
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...

    def gpu(self) -> 'Tensor':
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...

    def size(self, dim=None) -> Union[tuple, int]:
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if dim is None:
            return self._data.shape
        return self._data.shape[dim]

    def dim(self) -> int:
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return len(self._data.shape)

    @property
    def shape(self) -> tuple:
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return self._data.shape

    @property
    def ndim(self) -> int:
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return len(self._data.shape)

    @property
    def dtype(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return self._data.dtype

    @property
    def device(self) -> str:
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return self._device

    @property
    def data(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return self._data

    @property
    def is_leaf(self) -> bool:
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if not self._requires_grad:
            return True
        return self._grad_fn is None

    @property
    def grad(self) -> 'Tensor':
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if not self._can_read_grad():
            warnings.warn(
                'The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad '
                "attribute won't be populated during autograd.backward(). If you indeed want the gradient "
                'for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the '
                'non-leaf Tensor by mistake, make sure you access the leaf Tensor instead.',
                UserWarning,
            )

        return self._grad

    @grad.setter
    def grad(self, grad: 'Tensor'):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self._grad = grad

    @property
    def requires_grad(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, requires_grad):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if not requires_grad and not self.is_leaf:
            raise RuntimeError(
                'you cannot set requires grad to False when the tensor is not a leaf tensor, you need to detach the tensor from graph'
            )
        # if self.dtype != 'float32':
        #     raise RuntimeError('only float tensors can be required grad')
        self._requires_grad = requires_grad

    def retain_grad(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if not self.requires_grad:
            raise RuntimeError("can't retain_grad on Tensor that has requires_grad=False")
        if self.is_leaf:  # no-op for leaves
            return
        if hasattr(self, 'retains_grad'):
            return

        import weakref

        weak_self = weakref.ref(self)

        def retain_grad_hook(grad):
            var = weak_self()
            if var is None:
                return
            if var._grad is None:
                var._grad = grad.clone()
                # if grad.is_sparse:
                #     var._grad = grad.clone()
                # else:
                #     var._grad = grad.clone(memory_format=torch.contiguous_format)
            else:
                var._grad = var._grad + grad

        self.register_hook(retain_grad_hook)

        self.retains_grad = True

    def register_hook(self, hook):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if not self.requires_grad:
            raise RuntimeError("cannot register a hook on a tensor that doesn't require gradient")
        if self._backward_hooks is None:
            self._backward_hooks = OrderedDict()
            # if self._grad_fn is not None:
            #     self._grad_fn._register_hook_dict(self)
        handle = RemovableHandle(self._backward_hooks)
        self._backward_hooks[handle.id] = hook
        return handle

    def _can_read_grad(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return not (
                self.requires_grad and not hasattr(self, 'retains_grad') and not self.is_leaf and self._grad is None
        )

    def _can_write_grad(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return not (self.requires_grad and not hasattr(self, 'retains_grad') and not self.is_leaf)

    def forward(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """

    def backward(self, gradient=None):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        # if graph does not have any tensor that can have grad, RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn

        import numpy as np

        if gradient is None:
            gradient = Tensor.from_array(np.array(1))

        if 'numpy' in type(gradient).__module__:
            gradient = Tensor.from_array(np.array(np.copy(gradient)))

        if self.device == 'gpu':
            gradient.gpu()

        if self._grad_fn is not None:
            self._grad_fn(gradient=gradient)

        if self._backward_hooks is not None:
            for _backward_hook in self._backward_hooks.values():
                _backward_hook(gradient)


class Parameter(Tensor):
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

    def __init__(self, data=None, requires_grad=True):
        if data is None:
            data = Tensor()

        super(Parameter, self).__init__(data=data.data, requires_grad=requires_grad)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return 'Parameter containing:\n' + super(Parameter, self).__str__()


class Module:
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

    dump_patches: bool = False
    _version: int = 1
    training: bool

    def __init__(self):
        self.training = True
        self._parameters = OrderedDict()
        self._modules = OrderedDict()

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if '_parameters' not in self.__dict__:
            raise AttributeError('cannot assign parameter before Module.__init__() call')
        elif not isinstance(name, six.string_types):
            raise TypeError('parameter name should be a string. Got {}'.format(typename(name)))
        elif '.' in name:
            raise KeyError('parameter name can\'t contain "."')
        elif name == '':
            raise KeyError('parameter name can\'t be empty string ""')
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError("attribute '{}' already exists".format(name))

        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError(
                "cannot assign '{}' object to parameter '{}' (Parameter or None required)".format(typename(param), name)
            )
        elif not param.is_leaf:
            raise ValueError(
                "Cannot assign non-leaf Tensor to parameter '{0}'. Model parameters must be created explicitly. "
                "To express '{0}' as a function of another Tensor, compute the value in the forward() method.".format(
                    name
                )
            )
        else:
            self._parameters[name] = param

    def add_module(self, name: str, module: Optional['Module']) -> None:
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if not isinstance(module, Module) and module is not None:
            raise TypeError('{} is not a Module subclass'.format(typename(module)))
        elif not isinstance(name, six.string_types):
            raise TypeError('module name should be a string. Got {}'.format(typename(name)))
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError("attribute '{}' already exists".format(name))
        elif '.' in name:
            raise KeyError('module name can\'t contain "."')
        elif name == '':
            raise KeyError('module name can\'t be empty string ""')
        self._modules[name] = module

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getattr__(self, name: str) -> Union[Parameter, 'Module']:
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        raise ModuleAttributeException("'{}' object has no attribute '{}'".format(type(self).__name__, name))

    def __setattr__(self, name: str, value: Union[Parameter, 'Module']) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        params = self.__dict__.get('_parameters')
        # print(params)
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError('cannot assign parameters before Module.__init__() call')
            remove_from(self.__dict__, self._modules)
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError(
                    "cannot assign '{}' as parameter '{}' (Parameter or None expected)".format(typename(value), name)
                )
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get('_modules')
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError('cannot assign module before Module.__init__() call')
                remove_from(self.__dict__, self._parameters)
                modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError(
                        "cannot assign '{}' as child module '{}' (Module or None expected)".format(
                            typename(value), name
                        )
                    )
                modules[name] = value
            else:
                object.__setattr__(self, name, value)

    forward: Callable[..., Any] = _forward_unimplemented

    def _call_impl(self, *inp, **kwargs):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return self.forward(*inp, **kwargs)

    __call__: Callable[..., Any] = _call_impl

    def __delattr__(self, name):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._modules:
            del self._modules[name]
        else:
            object.__delattr__(self, name)

    def _save_to_state_dict(self, destination, prefix):
        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param.detach().to_array().tolist()

    def state_dict(self, *, destination=None, prefix=''):
        if destination is None:
            destination = OrderedDict()

        self._save_to_state_dict(destination, prefix)
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination=destination, prefix=prefix + name + '.')
        return destination

    def _load_from_state_dict(self, state_dict, prefix):
        local_name_params = self._parameters.items()
        local_state = {k: v for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = Tensor.from_array(state_dict[key])

                if len(param.shape) == 0 and len(input_param.shape) == 1:
                    input_param = input_param[0]

                if input_param.shape != param.shape:
                    # local shape should match the one in checkpoint
                    warnings.warn(
                        'size mismatch for {}: copying a param with shape {} from checkpoint, '
                        'the shape in current model is {}.'
                        .format(key, input_param.shape, param.shape)
                    )
                    continue

                try:
                    # Shape checks are already done above
                    if isinstance(param, Parameter) and not isinstance(input_param, Parameter):
                        setattr(self, name, Parameter(input_param))
                    else:
                        setattr(self, name, input_param)
                except Exception as ex:
                    warnings.warn(
                        f'While copying the parameter named "{key}", '
                        f'whose dimensions in the model are {param.size()} and '
                        f'whose dimensions in the checkpoint are {input_param.size()}, '
                        f'an exception occurred : {ex.args}.'
                    )

    def load_state_dict(self, state_dict: Mapping[str, Any]):
        if not isinstance(state_dict, Mapping):
            raise TypeError(f"Expected state_dict to be dict-like, got {type(state_dict)}.")

        # def load(module, local_state_dict, prefix=''):
        #     module._load_from_state_dict(local_state_dict, prefix)
        #     for name, child in module._modules.items():
        #         if child is not None:
        #             child_prefix = prefix + name + '.'
        #             child_state_dict = {k: v for k, v in local_state_dict.items() if k.startswith(child_prefix)}
        #             load(child, child_state_dict, child_prefix)

        # load(self, state_dict)
        # del load

    def _named_members(self, get_members_fn, prefix='', recurse=True):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        gen = self._named_members(lambda module: module._parameters.items(), prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def children(self) -> Iterator['Module']:
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        for name, module in self.named_children():
            yield module

    def named_children(self) -> Iterator[Tuple[str, 'Module']]:
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def modules(self) -> Iterator['Module']:
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        for name, module in self.named_modules():
            yield module

    def named_modules(self, memo: Optional[Set['Module']] = None, prefix: str = ''):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_modules(memo, submodule_prefix):
                    yield m

    # load_state_dict, save_state_dict, state_dict

    def train(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.training = True
        for module in self.children():
            module.train()
        return self

    def eval(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.training = False
        for module in self.children():
            module.eval()
        return self

    def zero_grad(self, set_to_none: bool = False):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        for p in self.parameters():
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    if p.grad._grad_fn is not None:
                        p.grad.detach(inplace=True)
                    else:
                        p.grad.requires_grad = False
                    p.grad.fill(0)

    def _apply(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """

    def apply(self, func: Callable[['Module'], None]):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        for module in self.children():
            module.apply(func)
        func(self)
        return self

    def half(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        for param in self.parameters():
            param.half()
        return self

    def single(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        for param in self.parameters():
            param.single()
        return self

    def double(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        for param in self.parameters():
            param.double()
        return self

    def cpu(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        for param in self.parameters():
            param.cpu()
        return self

    def gpu(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        for param in self.parameters():
            param.gpu()
        return self

    def _get_name(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return self.__class__.__name__

    def extra_repr(self) -> str:
        r"""Set the extra representation of the module

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        return ''

    def __repr__(self):
        extra_lines = []
        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = indent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str

    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._parameters.keys())
        modules = list(self._modules.keys())
        keys = module_attrs + attrs + parameters + modules

        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)


class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx: Union[slice, int]):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx: int, module: Module):
        key: str = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx: Union[slice, int]):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)
        # To preserve numbering
        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = OrderedDict(list(zip(str_indices, self._modules.values())))

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    # NB: We can't really type check this function as the type of input
    # may change dynamically (as is tested in
    # TestScript.test_sequential_intermediary_types).  Cannot annotate
    # with Any as TorchScript expects a more precise type
    def forward(self, inp):
        for module in self:
            inp = module(inp)
        return inp


class Loss(Module):
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

    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class Sequential(Module):
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

    def __init__(self, *args: Any):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx: int, module: Module) -> None:
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx: Union[slice, int]) -> None:
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self) -> int:
        return len(self._modules)

    def __dir__(self):
        keys = super(Sequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def forward(self, inp):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        for module in self:
            inp = module(inp)
        return inp


class Optimizer(object):
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

    def __init__(self, params, defaults):
        self.defaults = defaults

        if isinstance(params, Parameter):
            raise TypeError(
                'params argument given to the optimizer should be an iterable of Tensors or dicts, but got '
                + typename(params)
            )

        self.state = defaultdict(dict)
        self.param_groups = []

        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError('optimizer got an empty parameter list')
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

    def __getstate__(self):
        return {
            'defaults': self.defaults,
            'state': self.state,
            'param_groups': self.param_groups,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        for i, group in enumerate(self.param_groups):
            format_string += '\n'
            format_string += 'Parameter Group {0}\n'.format(i)
            for key in sorted(group.keys()):
                if key != 'params':
                    format_string += '    {0}: {1}\n'.format(key, group[key])
        format_string += ')'
        return format_string

    def zero_grad(self, set_to_none: bool = False):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad._grad_fn is not None:
                            p.grad.detach(inplace=True)
                        else:
                            p.grad._requires_grad = False
                        p.grad.zero()

    def step(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        raise NotImplementedError

    def add_param_group(self, param_group):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        assert isinstance(param_group, dict), 'param group must be a dict'

        params = param_group['params']
        if isinstance(params, Parameter):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError(
                'optimizer parameters need to be organized in ordered collections, but '
                'the ordering of tensors in sets will change between runs. Please use a list instead.'
            )
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not isinstance(param, Parameter):
                raise TypeError('optimizer can only optimize Tensors, but one of the params is ' + typename(param))
            if not param.is_leaf:
                raise ValueError("can't optimize a non-leaf Tensor")

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError("parameter group didn't specify a value of required optimization parameter " + name)
            else:
                param_group.setdefault(name, default)

        params = param_group['params']
        if len(params) != len(set(params)):
            warnings.warn(
                'optimizer contains a parameter group with duplicate parameters; '
                'in future, this will cause an error',
                stacklevel=3,
            )

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError('some parameters appear in more than one parameter group')

        self.param_groups.append(param_group)


class LRScheduler(object):
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

    def __init__(self, optimizer, last_epoch=-1, verbose=False):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))
        self.optimizer = optimizer

        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError(
                        "param 'initial_lr' is not specified in param_groups[{}] when resuming an optimizer".format(i)
                    )
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = last_epoch

        def with_counter(method):
            if getattr(method, '_with_counter', False):
                return method

            instance_ref = weakref.ref(method.__self__)
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0
        self._step_count = 0
        self.verbose = verbose

        self.step()

    def state_dict(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.__dict__.update(state_dict)

    def get_last_lr(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return self._last_lr

    def get_lr(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError

    def print_lr(self, is_verbose, group, lr, epoch=None):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if is_verbose:
            if epoch is None:
                print('Adjusting learning rate of group {} to {:.4e}.'.format(group, lr))
            else:
                print('Epoch {:5d}: adjusting learning rate of group {} to {:.4e}.'.format(epoch, group, lr))

    def step(self, epoch=None):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        # Raise a warning if old pattern is detected
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, '_with_counter'):
                warnings.warn(
                    'Seems like `optimizer.step()` has been overridden after learning rate scheduler '
                    'initialization. Please, make sure to call `optimizer.step()` before '
                    '`lr_scheduler.step()`.',
                    UserWarning,
                )

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn(
                    'Detected call of `lr_scheduler.step()` before `optimizer.step()`. '
                    'In PyTorch 1.1.0 and later, you should call them in the opposite order: '
                    '`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this '
                    'will result in PyTorch skipping the first value of the learning rate schedule.',
                    UserWarning,
                )
        self._step_count += 1

        class _enable_get_lr_call:
            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False

        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_lr()
            else:
                warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
                self.last_epoch = epoch
                if hasattr(self, '_get_closed_form_lr'):
                    values = self._get_closed_form_lr()
                else:
                    values = self.get_lr()

        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data
            param_group['lr'] = lr
            self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
