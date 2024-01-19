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
    Exception raised for errors in the module attribute.

    This class inherits from the built-in `AttributeError` class and does not introduce new methods or attributes.

    It is used when an attribute reference or assignment fails in the context of a module.
    """


class RemovableHandle(object):
    """
    A handle which provides the capability to remove a hook.

    This class is used to manage hooks which are added to modules. It provides a mechanism to remove a hook by its ID.

    # Attributes
        id (int): The unique identifier for the hook.
        next_id (int): The identifier to be assigned to the next hook.
        hooks_dict_ref (weakref): A weak reference to the dictionary storing the hooks.
    """
    id: int
    next_id: int = 0

    def __init__(self, hooks_dict: Any) -> None:
        """
        Initializes the RemovableHandle class.

        # Arguments
            hooks_dict (Any): The dictionary storing the hooks.
        """
        self.hooks_dict_ref = weakref.ref(hooks_dict)
        self.id = RemovableHandle.next_id
        RemovableHandle.next_id += 1

    def remove(self) -> None:
        """
        Removes the hook associated with this handle from the hooks dictionary.

        If the hooks dictionary still exists and contains the hook associated with this handle, the hook is removed.
        """
        hooks_dict = self.hooks_dict_ref()
        if hooks_dict is not None and self.id in hooks_dict:
            del hooks_dict[self.id]

    def __getstate__(self):
        """
        Returns the current state of the RemovableHandle instance for serialization.

        # Returns
            tuple: The weak reference to the hooks dictionary and the ID of the handle.
        """
        return self.hooks_dict_ref(), self.id

    def __setstate__(self, state) -> None:
        """
        Sets the state of the RemovableHandle instance during deserialization.

        # Arguments
            state (tuple): The weak reference to the hooks dictionary and the ID of the handle.
        """
        if state[0] is None:
            # create a dead reference
            self.hooks_dict_ref = weakref.ref(OrderedDict())
        else:
            self.hooks_dict_ref = weakref.ref(state[0])
        self.id = state[1]
        RemovableHandle.next_id = max(RemovableHandle.next_id, self.id + 1)

    def __enter__(self) -> 'RemovableHandle':
        """
        Implements the context management protocol.

        # Returns
            RemovableHandle: The instance of the RemovableHandle.
        """
        return self

    def __exit__(self, type: Any, value: Any, tb: Any) -> None:
        """
        Implements the context management protocol.

        This method is called when the `with` statement is exited. It ensures that the hook is removed when the handle is no longer in use.

        # Arguments
            type (Any): The type of the exception that caused the context to be exited. If the context was exited without an exception, this will be None.
            value (Any): The instance of the exception that caused the context to be exited. If the context was exited without an exception, this will be None.
            tb (Any): The traceback of the exception that caused the context to be exited. If the context was exited without an exception, this will be None.
        """
        self.remove()


class Tensor:
    """
    This class represents a Tensor, a multi-dimensional array used in deep learning computations.

    Attributes:
        _data: The actual data of the tensor.
        _requires_grad: A boolean indicating whether the tensor requires gradients.
        _grad_fn: The gradient function associated with the tensor.
        _grad: The gradient of the tensor.
        _name: The name of the tensor.
        _backward_hooks: A dictionary of backward hooks.
        _device: The device where the tensor is located ('cpu' or 'gpu').

    Methods:
        __repr__: Returns a string representation of the tensor.
        __str__: Returns a string representation of the tensor.
        __hash__: Returns the hash of the tensor.
        __getitem__: Returns a tensor for a given index.
        __setitem__: Sets the value of the tensor at a given index.
        __ge__: Checks if the tensor is greater than or equal to another tensor.
        __gt__: Checks if the tensor is greater than another tensor.
        __le__: Checks if the tensor is less than or equal to another tensor.
        __lt__: Checks if the tensor is less than another tensor.
        __eq__: Checks if the tensor is equal to another tensor.
        __ne__: Checks if the tensor is not equal to another tensor.
        __add__: Adds another tensor to the current tensor.
        __radd__: Adds another tensor to the current tensor.
        __sub__: Subtracts another tensor from the current tensor.
        __rsub__: Subtracts the current tensor from another tensor.
        __mul__: Multiplies the current tensor with another tensor.
        __rmul__: Multiplies the current tensor with another tensor.
        __truediv__: Divides the current tensor by another tensor.
        __pow__: Raises the current tensor to the power of another tensor.
        __abs__: Returns the absolute value of the tensor.
        __neg__: Returns the negative of the tensor.
        chunk: Splits the tensor into a specific number of chunks.
        view: Returns a new tensor with the same data but different size.
        index_select: Returns a new tensor which indexes the input tensor along dimension dim using the entries in index.
        zero: Fills the tensor with zeros.
        one: Fills the tensor with ones.
        fill: Fills the tensor with a scalar value.
        squeeze: Returns a tensor with all the dimensions of input of size 1 removed.
        expand_dim: Expands the shape of the tensor.
        transpose: Transposes the tensor.
        absolute: Returns the absolute value of the tensor.
        around: Evenly round to the given number of decimals.
        floor: Returns the largest integer less than or equal to each element.
        ceil: Returns the smallest integer greater than or equal to each element.
        clip: Clips (limits) the values in an array.
        negative: Returns the negative of the tensor.
        log: Returns the natural logarithm of the tensor.
        summation: Returns the sum of all elements in the input tensor.
        mean: Returns the mean of all elements in the input tensor.
        std: Returns the standard deviation of all elements in the input tensor.
        var: Returns the variance of all elements in the input tensor.
        add: Adds another tensor to the current tensor.
        sub: Subtracts another tensor from the current tensor.
        mul: Multiplies the current tensor with another tensor.
        div: Divides the current tensor by another tensor.
        power: Raises the current tensor to the power of another tensor.
        clone: Returns a copy of the tensor.
        detach: Detaches the tensor from the computation graph.
        from_array: Creates a tensor from a numpy array.
        to_array: Converts the tensor to a numpy array.
        half: Converts the tensor to half precision.
        single: Converts the tensor to single precision.
        double: Converts the tensor to double precision.
        cpu: Moves the tensor to the CPU.
        gpu: Moves the tensor to the GPU.
        size: Returns the size of the tensor.
        dim: Returns the number of dimensions of the tensor.
        shape: Returns the shape of the tensor.
        ndim: Returns the number of dimensions of the tensor.
        dtype: Returns the data type of the tensor.
        device: Returns the device of the tensor.
        data: Returns the data of the tensor.
        is_leaf: Checks if the tensor is a leaf node.
        grad: Returns the gradient of the tensor.
        requires_grad: Checks if the tensor requires gradients.
        retain_grad: Retains the gradient of the tensor.
        register_hook: Registers a backward hook.
        _can_read_grad: Checks if the gradient can be read.
        _can_write_grad: Checks if the gradient can be written.
        forward: Forward pass of the tensor.
        backward: Backward pass of the tensor.
    """

    def __init__(self, data=None, requires_grad=None):
        """
        Initializes the Tensor with the given data and requires_grad flag.

        Args:
            data: The actual data of the tensor.
            requires_grad: A boolean indicating whether the tensor requires gradients.
        """
        self._data = data
        self._requires_grad = requires_grad
        self._grad_fn = None
        self._grad = None
        self._name = None
        self._backward_hooks = None
        self._device = 'cpu'

    def __repr__(self):
        """
        Returns a string representation of the tensor.

        Returns:
            str: A string representation of the tensor.
        """
        return str(self)

    def __str__(self):
        """
        Returns a string representation of the tensor.

        Returns:
            str: A string representation of the tensor.
        """
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
        """
        Returns the hash of the tensor.

        Returns:
            int: The hash of the tensor.
        """
        return id(self)

    def __getitem__(self, item):
        """
        Returns a tensor for a given index.

        Args:
            item: The index.

        Returns:
            Tensor: The tensor at the given index.
        """
        data = self.data[item]
        tensor = Tensor.from_array(data, requires_grad=self.requires_grad)
        tensor._grad_fn = self._grad_fn
        if self.device == 'gpu':
            tensor = tensor.gpu()
        return tensor

    def __setitem__(self, key, value):
        """
        Sets the value of the tensor at a given index.

        Args:
            key: The index.
            value: The value to set.
        """
        # self._grad_fn = value._grad_fn
        # self._requires_grad = value.requires_grad
        self._data[key] = value.data

    def __ge__(self, other):
        """
        Checks if the tensor is greater than or equal to another tensor.

        Args:
            other: The other tensor.

        Returns:
            bool: True if the tensor is greater than or equal to the other tensor, False otherwise.
        """
        # return self.greater_or_equal(other)
        if isinstance(other, Tensor):
            return Tensor(data=self.data >= other.data)
        else:
            return Tensor(data=self.data >= other)

    def __gt__(self, other):
        """
        Checks if the tensor is greater than another tensor.

        Args:
            other: The other tensor.

        Returns:
            bool: True if the tensor is greater than the other tensor, False otherwise.
        """
        # return self.greater(other)
        if isinstance(other, Tensor):
            return Tensor(data=self.data > other.data)
        else:
            return Tensor(data=self.data > other)

    def __le__(self, other):
        """
        Checks if the tensor is less than or equal to another tensor.

        Args:
            other: The other tensor.

        Returns:
            bool: True if the tensor is less than or equal to the other tensor, False otherwise.
        """
        # return self.lesser_or_equal(other)
        if isinstance(other, Tensor):
            return Tensor(data=self.data <= other.data)
        else:
            return Tensor(data=self.data <= other)

    def __lt__(self, other):
        """
        Checks if the tensor is less than another tensor.

        Args:
            other: The other tensor.

        Returns:
            bool: True if the tensor is less than the other tensor, False otherwise.
        """
        # return self.lesser(other)
        if isinstance(other, Tensor):
            return Tensor(data=self.data < other.data)
        else:
            return Tensor(data=self.data < other)

    def __eq__(self, other):
        """
        Checks if the tensor is equal to another tensor.

        Args:
            other: The other tensor.

        Returns:
            bool: True if the tensor is equal to the other tensor, False otherwise.
        """
        # return self.equal(other)
        if isinstance(other, Tensor):
            return Tensor(data=self.data == other.data)
        else:
            return Tensor(data=self.data == other)

    def __ne__(self, other):
        """
        Checks if the tensor is not equal to another tensor.

        Args:
            other: The other tensor.

        Returns:
            bool: True if the tensor is not equal to the other tensor, False otherwise.
        """
        # return self.not_equal(other)
        if isinstance(other, Tensor):
            return Tensor(data=self.data != other.data)
        else:
            return Tensor(data=self.data != other)

    def __add__(self, other) -> 'Tensor':
        """
        Adds another tensor to the current tensor.

        Args:
            other: The other tensor.

        Returns:
            Tensor: The result of the addition.
        """
        return self.add(other)

    def __radd__(self, other):
        """
        Adds another tensor to the current tensor.

        Args:
            other: The other tensor.

        Returns:
            Tensor: The result of the addition.
        """
        return self.add(other)

    def __sub__(self, other) -> 'Tensor':
        """
        Subtracts another tensor from the current tensor.

        Args:
            other: The other tensor.

        Returns:
            Tensor: The result of the subtraction.
        """
        return self.sub(other)

    def __rsub__(self, other):
        """
        Subtracts the current tensor from another tensor.

        Args:
            other: The other tensor.

        Returns:
            Tensor: The result of the subtraction.
        """
        return -self + other

    def __mul__(self, other) -> 'Tensor':
        """
        Implements the multiplication operation for the Tensor object.

        Args:
            other: The other operand involved in the multiplication operation.

        Returns:
            A new Tensor object that is the result of the multiplication operation.
        """
        return self.mul(other)

    def __rmul__(self, other):
        """
        Implements the right multiplication operation for the Tensor object.

        Args:
            other: The other operand involved in the multiplication operation.

        Returns:
            A new Tensor object that is the result of the multiplication operation.
        """
        return self.mul(other)

    def __truediv__(self, other) -> 'Tensor':
        """
        Implements the true division operation for the Tensor object.

        Args:
            other: The other operand involved in the division operation.

        Returns:
            A new Tensor object that is the result of the division operation.
        """
        return self.div(other)

    def __pow__(self, power, modulo=None) -> 'Tensor':
        """
        Implements the power operation for the Tensor object.

        Args:
            power: The power to which the Tensor object is to be raised.
            modulo: Not used in this method, present for compatibility with Python's dunder method requirements.

        Returns:
            A new Tensor object that is the result of the power operation.
        """
        return self.power(power)

    def __abs__(self) -> 'Tensor':
        """
        Implements the absolute operation for the Tensor object.

        Returns:
            A new Tensor object that is the absolute value of the original Tensor object.
        """
        return self.absolute()

    def __neg__(self):
        """
        Implements the negation operation for the Tensor object.

        Returns:
            A new Tensor object that is the negation of the original Tensor object.
        """
        return self.negative()

    def chunk(self, chunks, dim=0):
        """
        Splits the Tensor into a specific number of chunks.

        Args:
            chunks (int): The number of chunks to split the Tensor into.
            dim (int, optional): The dimension along which to split the Tensor. Defaults to 0.

        Returns:
            A list of Tensor objects which are chunks of the original Tensor.
        """

    def view(self, size) -> 'Tensor':
        """
        Returns a new tensor with the same data but different size.

        Args:
            size (tuple): The desired size.

        Returns:
            A new Tensor object with the desired size.
        """

    def index_select(self, dim, index) -> 'Tensor':
        """
        Returns a new tensor which indexes the input tensor along dimension dim using the entries in index.

        Args:
            dim (int): The dimension in which to index.
            index (Tensor): A tensor containing the indices to select.

        Returns:
            A new Tensor which indexes the original Tensor along dimension dim using the entries in index.
        """

    def zero(self) -> 'Tensor':
        """
        Fills the Tensor with zeros.

        Returns:
            The original Tensor filled with zeros.
        """

    def one(self) -> 'Tensor':
        """
        Fills the Tensor with ones.

        Returns:
            The original Tensor filled with ones.
        """

    def fill(self, value) -> 'Tensor':
        """
        Fills the Tensor with the specified value.

        Args:
            value: The value to fill the Tensor with.

        Returns:
            The original Tensor filled with the specified value.
        """

    def squeeze(self, axis=None) -> 'Tensor':
        """
        Removes dimensions of size one from the Tensor.

        Args:
            axis (int, optional): The specific dimension to remove. If None, all dimensions of size one will be removed.

        Returns:
            A new Tensor with the dimensions of size one removed.
        """

    def expand_dim(self, axis=None) -> 'Tensor':
        """
        Expands the dimensions of the Tensor.

        Args:
            axis (int, optional): The dimension to expand. If None, all dimensions will be expanded.

        Returns:
            A new Tensor with the dimensions expanded.
        """

    def transpose(self, axes) -> 'Tensor':
        """
        Transposes the Tensor.

        Args:
            axes (tuple): The axes along which to transpose the Tensor.

        Returns:
            A new Tensor that is the transpose of the original Tensor.
        """

    def absolute(self) -> 'Tensor':
        """
        Computes the absolute values of the Tensor.

        Returns:
            A new Tensor that is the absolute value of the original Tensor.
        """

    def around(self) -> 'Tensor':
        """
        Rounds the Tensor to the nearest integer.

        Returns:
            A new Tensor that is the rounded version of the original Tensor.
        """

    def floor(self) -> 'Tensor':
        """
        Rounds the Tensor down to the nearest integer.

        Returns:
            A new Tensor that is the floor of the original Tensor.
        """

    def ceil(self) -> 'Tensor':
        """
        Rounds the Tensor up to the nearest integer.

        Returns:
            A new Tensor that is the ceiling of the original Tensor.
        """

    def clip(self, min_val, max_val) -> 'Tensor':
        """
        Clips the Tensor to be within a specified range.

        Args:
            min_val: The minimum value for the Tensor.
            max_val: The maximum value for the Tensor.

        Returns:
            A new Tensor that is the original Tensor clipped to be within the specified range.
        """

    def negative(self) -> 'Tensor':
        """
        Computes the negative of the Tensor.

        Returns:
            A new Tensor that is the negative of the original Tensor.
        """

    def log(self) -> 'Tensor':
        """
        Computes the natural logarithm of the Tensor.

        Returns:
            A new Tensor that is the natural logarithm of the original Tensor.
        """

    def summation(self) -> 'Tensor':
        """
        Computes the sum of all elements in the Tensor.

        Returns:
            A new Tensor that is the sum of all elements in the original Tensor.
        """

    def mean(self) -> 'Tensor':
        """
        Computes the mean of all elements in the Tensor.

        Returns:
            A new Tensor that is the mean of all elements in the original Tensor.
        """

    def std(self) -> 'Tensor':
        """
        Computes the standard deviation of all elements in the Tensor.

        Returns:
            A new Tensor that is the standard deviation of all elements in the original Tensor.
        """

    def var(self) -> 'Tensor':
        """
        Computes the variance of all elements in the Tensor.

        Returns:
            A new Tensor that is the variance of all elements in the original Tensor.
        """

    def add(self, other) -> 'Tensor':
        """
        Adds another Tensor to the current Tensor.

        Args:
            other: The other Tensor to add.

        Returns:
            A new Tensor that is the result of the addition operation.
        """

    def sub(self, other) -> 'Tensor':
        """
        Subtracts another Tensor from the current Tensor.

        Args:
            other: The other Tensor to subtract.

        Returns:
            A new Tensor that is the result of the subtraction operation.
        """

    def mul(self, other) -> 'Tensor':
        """
        Multiplies the current Tensor by another Tensor.

        Args:
            other: The other Tensor to multiply.

        Returns:
            A new Tensor that is the result of the multiplication operation.
        """

    def div(self, other) -> 'Tensor':
        """
        Divides the current Tensor by another Tensor.

        Args:
            other: The other Tensor to divide.

        Returns:
            A new Tensor that is the result of the division operation.
        """

    def power(self, p) -> 'Tensor':
        """
        Raises the current Tensor to the power of p.

        Args:
            p: The power to raise the Tensor to.

        Returns:
            A new Tensor that is the result of the power operation.
        """

    def clone(self) -> 'Tensor':
        """
        Creates a copy of the current Tensor.

        Returns:
            A new Tensor that is a copy of the current Tensor.
        """

    def detach(self, inplace=False) -> 'Tensor':
        """
        Detaches the Tensor from the computation graph.

        Args:
            inplace (bool, optional): If True, the operation is performed in-place. Defaults to False.

        Returns:
            The detached Tensor.
        """

    @staticmethod
    def from_array(data, requires_grad=False) -> 'Tensor':
        """
        Creates a Tensor from a numpy array.

        Args:
            data: The numpy array to convert into a Tensor.
            requires_grad (bool, optional): If True, the Tensor will require gradient computation. Defaults to False.

        Returns:
            A new Tensor created from the numpy array.
        """

    def to_array(self):
        """
        Converts the Tensor into a numpy array.

        Returns:
            A numpy array that is a copy of the Tensor data.
        """

    def half(self) -> 'Tensor':
        """
        Converts the Tensor to half precision.

        Returns:
            A new Tensor that is the half precision version of the original Tensor.
        """

    def single(self) -> 'Tensor':
        """
        Converts the Tensor to single precision.

        Returns:
            A new Tensor that is the single precision version of the original Tensor.
        """

    def double(self) -> 'Tensor':
        """
        Converts the Tensor to double precision.

        Returns:
            A new Tensor that is the double precision version of the original Tensor.
        """

    def cpu(self) -> 'Tensor':
        """
        Moves the Tensor to the CPU.

        Returns:
            The Tensor after it has been moved to the CPU.
        """

    def gpu(self) -> 'Tensor':
        """
        Moves the Tensor to the GPU.

        Returns:
            The Tensor after it has been moved to the GPU.
        """

    def size(self, dim=None) -> Union[tuple, int]:
        """
        Returns the size of the Tensor.

        Args:
            dim (int, optional): If specified, the size of the specific dimension is returned. Otherwise, the size of all dimensions is returned.

        Returns:
            The size of the Tensor.
        """
        if dim is None:
            return self._data.shape
        return self._data.shape[dim]

    def dim(self) -> int:
        """
        Returns the number of dimensions of the Tensor.

        Returns:
            The number of dimensions of the Tensor.
        """
        return len(self._data.shape)

    @property
    def shape(self) -> tuple:
        """
        Returns the shape of the Tensor.

        Returns:
            The shape of the Tensor.
        """
        return self._data.shape

    @property
    def ndim(self) -> int:
        """
        Returns the number of dimensions of the Tensor.

        Returns:
            The number of dimensions of the Tensor.
        """
        return len(self._data.shape)

    @property
    def dtype(self):
        """
        Returns the data type of the Tensor's underlying data.

        # Returns
            The data type of the Tensor's underlying data.
        """
        return self._data.dtype

    @property
    def device(self) -> str:
        """
        Returns the device where the Tensor is located.

        # Returns
            A string representing the device where the Tensor is located. It could be 'cpu' or 'gpu'.
        """
        return self._device

    @property
    def data(self):
        """
        Returns the underlying data of the Tensor.

        # Returns
            The underlying data of the Tensor.
        """
        return self._data

    @property
    def is_leaf(self) -> bool:
        """
        Checks if the Tensor is a leaf node.

        A Tensor is considered a leaf if it was not the result of an operation.
        That is, if it was read from data or if it is a constant.

        # Returns
            True if the Tensor is a leaf node, False otherwise.
        """
        if not self._requires_grad:
            return True
        return self._grad_fn is None

    @property
    def grad(self) -> 'Tensor':
        """
        Returns the gradient of the Tensor.

        The gradient is computed with respect to some scalar value.

        # Returns
            The gradient of the Tensor.
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
        Sets the gradient of the Tensor.

        # Arguments
            grad (Tensor): The gradient to be set.
        """
        self._grad = grad

    @property
    def requires_grad(self):
        """
        Checks if the Tensor requires gradient computation.

        # Returns
            True if the Tensor requires gradient computation, False otherwise.
        """
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, requires_grad):
        """
        Sets the Tensor's requirement for gradient computation.

        # Arguments
            requires_grad (bool): Whether the Tensor should require gradient computation.
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
        Allows a non-leaf Tensor to retain its gradient.

        Normally, only leaf Tensors (those not resulting from an operation) will have their gradients retained.
        This method allows non-leaf Tensors to retain their gradients.
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
        Registers a backward hook.

        Backward hooks are functions that are executed every time a backward operation is performed.

        # Arguments
            hook (function): The backward hook function to register.
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
        Checks if the gradient of the Tensor can be read.

        # Returns
            True if the gradient can be read, False otherwise.
        """
        return not (
                self.requires_grad and not hasattr(self, 'retains_grad') and not self.is_leaf and self._grad is None
        )

    def _can_write_grad(self):
        """
        Checks if the gradient of the Tensor can be written.

        # Returns
            True if the gradient can be written, False otherwise.
        """
        return not (self.requires_grad and not hasattr(self, 'retains_grad') and not self.is_leaf)

    def forward(self):
        """
        Performs the forward pass of the Tensor.

        This method should be overridden by all subclasses.
        """

    def backward(self, gradient=None):
        """
        Performs the backward pass of the Tensor.

        Computes the gradient of the Tensor with respect to some scalar value.

        # Arguments
            gradient (Tensor, optional): The gradient of the subsequent layer in the computation graph.
                                         If None, a Tensor of ones, with the same shape as the current Tensor, is used.
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
    A Parameter is a kind of Tensor that is to be considered a module parameter.

    Parameters are Tensor subclasses, that have a very special property when used with Module s - when they’re assigned as Module attributes they are automatically added to the list of its parameters, and will appear in parameters() iterator. Assigning a Tensor doesn’t have such effect. This is because one might want to cache some temporary state (more on this later) in the model. If there was no such class as Parameter, these temporaries would get registered too.

    Another difference is that parameters can't be volatile and that they require gradient by default.

    Args:
        data (Tensor, optional): Parameter data. Default: None
        requires_grad (bool, optional): If the parameter requires gradient. Default: `True`
    """

    def __init__(self, data=None, requires_grad=True):
        """
        Initializes the Parameter class.

        Args:
            data (Tensor, optional): Parameter data. Default: None
            requires_grad (bool, optional): If the parameter requires gradient. Default: `True`
        """
        if data is None:
            data = Tensor()

        super(Parameter, self).__init__(data=data.data, requires_grad=requires_grad)

    def __repr__(self):
        """
        Returns a string representation of the Parameter.

        Returns:
            str: a string representation of the Parameter
        """
        return str(self)

    def __str__(self):
        """
        Returns a string representation of the Parameter.

        Returns:
            str: a string representation of the Parameter
        """
        return 'Parameter containing:\n' + super(Parameter, self).__str__()


class Module:
    """
    Base class for all neural network modules.

    Your models should also subclass this class.

    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes.

    Subclassed Modules should initialize the parent class with the arguments
    they receive.

    Attributes:
        dump_patches (bool): If True, the model will dump its state_dict() after each forward pass.
        _version (int): The version of the model.
        training (bool): If True, the model is in training mode. If False, the model is in evaluation mode.
    """

    dump_patches: bool = False
    _version: int = 1
    training: bool

    def __init__(self):
        """
        Initializes the Module class.

        Sets the model to training mode and initializes the parameters and modules as empty OrderedDicts.
        """
        self.training = True
        self._parameters = OrderedDict()
        self._modules = OrderedDict()

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        """
        Adds a parameter to the module.

        The parameter can be accessed as an attribute using the given name.

        Args:
            name (str): name of the parameter. The string should be a valid attribute name.
            param (Parameter, optional): parameter to be added to the module. The parameter should be an instance of the Parameter class.
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
        Adds a child module to the current module.

        The module can be accessed as an attribute using the given name.

        Args:
            name (str): name of the child module. The string should be a valid attribute name.
            module (Module, optional): child module to be added to the module. The module should be an instance of the Module class.
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
        """
        Returns the state of the module.

        This function is called when the module is being pickled.

        Returns:
            dict: a dictionary representing the state of the module.
        """
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """
        Sets the state of the module.

        This function is called when the module is being unpickled.

        Args:
            state (dict): a dictionary representing the state of the module.
        """
        self.__dict__.update(state)

    def __getattr__(self, name: str) -> Union[Parameter, 'Module']:
        """
        Returns the attribute of the module with the given name.

        If the attribute is a Parameter or a Module, it is returned as is.
        If the attribute does not exist, a ModuleAttributeException is raised.

        Args:
            name (str): name of the attribute.

        Returns:
            Parameter or Module: the attribute of the module.
        """
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
        """
        Sets the attribute of the module with the given name.

        If the attribute is a Parameter or a Module, it is added to the respective dictionary of the module.

        Args:
            name (str): name of the attribute.
            value (Parameter or Module): value of the attribute.
        """

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
        Calls the forward function of the module.

        This function is called when the module is called. It can be overridden in subclasses.

        Args:
            *inp: the input that was passed to the module call.
            **kwargs: the keyword arguments that were passed to the module call.

        Returns:
            The result of the forward function.
        """
        return self.forward(*inp, **kwargs)

    __call__: Callable[..., Any] = _call_impl

    def __delattr__(self, name):
        """
        Deletes the attribute of the module with the given name.

        Args:
            name (str): name of the attribute.
        """
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._modules:
            del self._modules[name]
        else:
            object.__delattr__(self, name)

    def _save_to_state_dict(self, destination, prefix):
        """
        Saves the state of the module to the destination dictionary.

        The state is saved under the key that is a combination of the prefix and the parameter name.

        Args:
            destination (dict): the dictionary to which the state should be saved.
            prefix (str): the prefix to be used for the key.
        """
        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param.detach().to_array().tolist()

    def state_dict(self, *, destination=None, prefix=''):
        """
        Returns a dictionary representing the state of the module.

        If a destination is provided, it is used. Otherwise, a new dictionary is created.

        Args:
            destination (dict, optional): the dictionary to which the state should be saved. If None, a new dictionary is created.
            prefix (str, optional): the prefix to be used for the keys in the state dictionary.

        Returns:
            dict: a dictionary representing the state of the module.
        """
        if destination is None:
            destination = OrderedDict()

        self._save_to_state_dict(destination, prefix)
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination=destination, prefix=prefix + name + '.')
        return destination

    def _load_from_state_dict(self, state_dict, prefix):
        """
        Loads the state of the module from the state_dict.

        The state is loaded from the keys that start with the prefix.

        Args:
            state_dict (dict): the dictionary from which the state should be loaded.
            prefix (str): the prefix of the keys from which the state should be loaded.
        """
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
        """
        Loads the state of the module from the state_dict.

        Args:
            state_dict (dict): the dictionary from which the state should be loaded.
        """
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
        Returns an iterator over module members.

        If recurse is True, then yields members of this module and all submodules.
        Otherwise, yields only members that are direct children of this module.

        Args:
            get_members_fn (callable): A callable that takes a module and returns an iterator over its members.
            prefix (str): Prefix to prepend to all member names.
            recurse (bool): If True, recurse over all submodules.

        Yields:
            (string, Module): Tuple containing a name and a module.
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
        Returns an iterator over module parameters.

        This is typically passed to an optimizer.

        Args:
            recurse (bool): If True, then yields parameters of this module and all submodules.
                            Otherwise, yields only parameters that are direct children of this module.

        Yields:
            Parameter: Module parameter.
        """
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        """
        Returns an iterator over module parameters, yielding both the name of the parameter
        as well as the parameter itself.

        Args:
            prefix (str, optional): Prefix for each parameter name. Defaults to ''.
            recurse (bool, optional): If True, includes parameters of this module and all submodules.
                                      If False, includes only parameters that are direct members of this module.
                                      Defaults to True.

        Yields:
            Iterator[Tuple[str, Parameter]]: An iterator over name, parameter pairs.
        """
        gen = self._named_members(lambda module: module._parameters.items(), prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def children(self) -> Iterator['Module']:
        """
        Returns an iterator over immediate child modules.

        Yields:
            Iterator['Module']: An iterator over child modules.
        """
        for name, module in self.named_children():
            yield module

    def named_children(self) -> Iterator[Tuple[str, 'Module']]:
        """
        Returns an iterator over immediate child modules, yielding both the name of the child module
        as well as the module itself.

        Yields:
            Iterator[Tuple[str, 'Module']]: An iterator over name, module pairs.
        """
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def modules(self) -> Iterator['Module']:
        """
        Returns an iterator over all modules in the current module.

        Yields:
            Iterator['Module']: An iterator over all modules.
        """
        for name, module in self.named_modules():
            yield module

    def named_modules(self, memo: Optional[Set['Module']] = None, prefix: str = ''):
        """
        Returns an iterator over all modules in the current module, yielding both the name of the module
        as well as the module itself.

        Args:
            memo (Optional[Set['Module']], optional): Set of modules already processed. Defaults to None.
            prefix (str, optional): Prefix for each module name. Defaults to ''.

        Yields:
            Iterator[Tuple[str, 'Module']]: An iterator over name, module pairs.
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
        Sets the module in training mode.

        Returns:
            self: The current module.
        """
        self.training = True
        for module in self.children():
            module.train()
        return self

    def eval(self):
        """
        Sets the module in evaluation mode.

        Returns:
            self: The current module.
        """
        self.training = False
        for module in self.children():
            module.eval()
        return self

    def zero_grad(self, set_to_none: bool = False):
        """
        Sets the gradients of all parameters to zero.

        Args:
            set_to_none (bool, optional): If True, instead of setting to zero, sets the gradients to None. Defaults to False.
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
        Internal method to apply a function to the Module.
        The function should modify the module in-place.
        This method is meant to be overridden by subclasses.
        """

    def apply(self, func: Callable[['Module'], None]):
        """
        Applies a function on all modules (self and children).

        Args:
            func (Callable[['Module'], None]): A function that will be applied to all modules.
        """
        for module in self.children():
            module.apply(func)
        func(self)
        return self

    def half(self):
        """
        Casts all parameters and buffers to half precision.

        Returns:
            self: The current module.
        """
        for param in self.parameters():
            param.half()
        return self

    def single(self):
        """
        Casts all parameters and buffers to single precision.

        Returns:
            self: The current module.
        """
        for param in self.parameters():
            param.single()
        return self

    def double(self):
        """
        Casts all parameters and buffers to double precision.

        Returns:
            self: The current module.
        """
        for param in self.parameters():
            param.double()
        return self

    def cpu(self):
        """
        Moves all parameters and buffers to the CPU.

        Returns:
            self: The current module.
        """
        for param in self.parameters():
            param.cpu()
        return self

    def gpu(self):
        """
        Moves all parameters and buffers to the GPU.

        Returns:
            self: The current module.
        """
        for param in self.parameters():
            param.gpu()
        return self

    def _get_name(self):
        """
        Returns the name of the module.

        Returns:
            str: The name of the module.
        """
        return self.__class__.__name__

    def extra_repr(self) -> str:
        """
        Sets the extra representation of the module.
        To print customized extra information, you should re-implement this method in your own modules.
        Both single-line and multi-line strings are acceptable.

        Returns:
            str: Extra representation string (empty by default).
        """
        return ''

    def __repr__(self):
        """
        Returns a string containing a brief representation of the module.

        Returns:
            str: Representation string of the module.
        """
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
        """
        Returns a list of attributes of this module.
        This method controls the behavior of dir() invoked on a module object.

        Returns:
            List[str]: List of attributes.
        """
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._parameters.keys())
        modules = list(self._modules.keys())
        keys = module_attrs + attrs + parameters + modules

        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)


class Sequential(Module):
    """
    A sequential container. Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    Args:
        *args (optional): an ordered list of modules.
    """

    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """
        Get the idx-th item of the iterator.

        Args:
            iterator: an iterator.
            idx: the index of the item to get.

        Returns:
            The idx-th item of the iterator.
        """
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx: Union[slice, int]):
        """
        Get the item at a given index.

        Args:
            idx (Union[slice, int]): the index of the item to get.

        Returns:
            The item at the given index.
        """
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx: int, module: Module):
        """
        Set the module at a given index.

        Args:
            idx (int): the index to set the module at.
            module (Module): the module to set.
        """
        key: str = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx: Union[slice, int]):
        """
        Delete the module at a given index.

        Args:
            idx (Union[slice, int]): the index of the module to delete.
        """
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
        """
        Get the number of modules in the container.

        Returns:
            The number of modules in the container.
        """
        return len(self._modules)

    def __iter__(self):
        """
        Get an iterator over the modules in the container.

        Returns:
            An iterator over the modules in the container.
        """
        return iter(self._modules.values())

    # NB: We can't really type check this function as the type of input
    # may change dynamically (as is tested in
    # TestScript.test_sequential_intermediary_types).  Cannot annotate
    # with Any as TorchScript expects a more precise type
    def forward(self, inp):
        """
        Defines the computation performed at every call.

        Args:
            inp: the input to the forward function.

        Returns:
            The output of the forward function.
        """
        for module in self:
            inp = module(inp)
        return inp


class Loss(Module):
    """
    This class represents a Loss module which is a subclass of the Module class.
    It is used to compute the loss value during the training of a neural network model.

    The class uses the concept of Deep Deterministic Policy Gradient for loss computation.

    Attributes:
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                         'none': no reduction will be applied.
                         'mean': the sum of the output will be divided by the number of elements in the output.
                         'sum': the output will be summed.

    Args:
        size_average (bool, optional): Deprecated (see reduction). By default, the losses are averaged over each loss element in the batch.
                                        Note that for some losses, there are multiple elements per sample.
                                        If the field size_average is set to False, the losses are instead summed for each minibatch.
                                        Ignored when reduce is False. Default: True
        reduce (bool, optional): Deprecated (see reduction). By default, the losses are averaged or summed over observations for each minibatch depending
                                  on size_average. When reduce is False, returns a loss per batch element instead and ignores size_average.
                                  Default: True
        reduction (str, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                                    'none': no reduction will be applied.
                                    'mean': the sum of the output will be divided by the number of elements in the output.
                                    'sum': the output will be summed.
                                    Default: 'mean'
    """

    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        """
        Initializes the Loss class.

        Args:
            size_average (bool, optional): Deprecated (see reduction). By default, the losses are averaged over each loss element in the batch.
                                            Note that for some losses, there are multiple elements per sample.
                                            If the field size_average is set to False, the losses are instead summed for each minibatch.
                                            Ignored when reduce is False. Default: True
            reduce (bool, optional): Deprecated (see reduction). By default, the losses are averaged or summed over observations for each minibatch depending
                                      on size_average. When reduce is False, returns a loss per batch element instead and ignores size_average.
                                      Default: True
            reduction (str, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                                        'none': no reduction will be applied.
                                        'mean': the sum of the output will be divided by the number of elements in the output.
                                        'sum': the output will be summed.
                                        Default: 'mean'
        """
        super(Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class Optimizer(object):
    """
    Base class for all optimizers.

    Optimizers are used to update the parameters of a model in order to minimize the loss function.

    Args:
        params (iterable): An iterable of Parameters that define what Tensors should be optimized.
        defaults (dict): A dictionary containing default values of optimization options (like learning rate, weight decay, etc).

    Attributes:
        defaults (dict): The default optimization options.
        state (dict): A dictionary that holds current optimization state. Its content
            differs between optimizer classes.
        param_groups (list): A list of parameter groups. Each group is a dictionary that
            holds parameters and their corresponding optimization options.
    """

    def __init__(self, params, defaults):
        """
        Initializes the Optimizer class.

        Args:
            params (iterable): An iterable of Parameters that define what Tensors should be optimized.
            defaults (dict): A dictionary containing default values of optimization options (like learning rate, weight decay, etc).
        """
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
        """
        Returns the state of the optimizer. Useful for serialization.

        Returns:
            dict: The state of the optimizer.
        """
        return {
            'defaults': self.defaults,
            'state': self.state,
            'param_groups': self.param_groups,
        }

    def __setstate__(self, state):
        """
        Sets the state of the optimizer. Useful for deserialization.

        Args:
            state (dict): The state of the optimizer.
        """
        self.__dict__.update(state)

    def __repr__(self):
        """
        Returns a string representation of the optimizer.

        Returns:
            str: The string representation of the optimizer.
        """
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
        Clears the gradients of all optimized Tensors.

        Args:
            set_to_none (bool, optional): Instead of filling with zero, sets the gradients to None.
                This will in general have lower memory footprint. Defaults to False.
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
        Performs a single optimization step (parameter update).

        Should be overridden by all subclasses.

        Note:
            It is recommended to use `torch.no_grad()` on the enclosing scope to disable
            gradient computation for performance.

        Raises:
            NotImplementedError: This method needs to be implemented in subclasses.
        """
        raise NotImplementedError

    def add_param_group(self, param_group):
        """
        Adds a parameter group to the optimizer’s param_groups.

        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the optimizer as training progresses.

        Args:
            param_group (dict): Specifies what Tensors should be optimized along with group-specific optimization options.
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
    Base class for learning rate schedulers.

    This class provides the basic structure for implementing different learning rate scheduling policies.
    Learning rate schedulers dynamically adjust the learning rate of the optimizer during the training process,
    which can lead to improved model performance and robustness.

    Args:
        optimizer (Optimizer): The optimizer for which the learning rate will be scheduled.
        last_epoch (int, optional): The index of the last epoch. Default: -1.
        verbose (bool, optional): If True, prints a message to stdout for each update. Default: False.
    """

    def __init__(self, optimizer, last_epoch=-1, verbose=False):
        """
        Initializes the LRScheduler class.

        Args:
            optimizer (Optimizer): The optimizer for which the learning rate will be scheduled.
            last_epoch (int, optional): The index of the last epoch. Default: -1.
            verbose (bool, optional): If True, prints a message to stdout for each update. Default: False.
        """
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
        Returns the state of the scheduler as a dictionary.

        It contains all items necessary to keep track of the scheduler's state.
        Note that it does not contain the state of the optimizer.

        Returns:
            dict: The state of the scheduler.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """
        Loads the schedulers state.

        Args:
            state_dict (dict): Scheduler state. Should be an object returned from a call to `state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_lr(self):
        """
        Returns last computed learning rate by scheduler.

        Returns:
            list: Last computed learning rate by scheduler.
        """
        return self._last_lr

    def get_lr(self):
        """
        Computes the learning rate at each step. Needs to be implemented by subclasses.

        Raises:
            NotImplementedError: This method needs to be implemented in subclasses.
        """
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError

    def print_lr(self, is_verbose, group, lr, epoch=None):
        """
        Prints the learning rate of a specific parameter group.

        Args:
            is_verbose (bool): If True, prints a message to stdout for each update.
            group (int): Index of the parameter group.
            lr (float): Learning rate of the parameter group.
            epoch (int, optional): Current epoch number.
        """
        if is_verbose:
            if epoch is None:
                print('Adjusting learning rate of group {} to {:.4e}.'.format(group, lr))
            else:
                print('Epoch {:5d}: adjusting learning rate of group {} to {:.4e}.'.format(epoch, group, lr))

    def step(self, epoch=None):
        """
        Updates the learning rate of the optimizer.

        Args:
            epoch (int, optional): Current epoch number. Default: None.
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
