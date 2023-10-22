import math
from functools import (
    partial,
    update_wrapper
)
from typing import (
    List,
    Tuple,
    Union
)

from joatmon.nn.core import Tensor
from joatmon.nn.utility import (
    _calculate_input_dims,
    _calculate_output_dims
)

BooleanT = Union[bool]
IntegerT = Union[int]
FloatingT = Union[float]
NumberT = Union[IntegerT, FloatingT]

DataT = Union[BooleanT, IntegerT, FloatingT]
ArrayT = Union[Tuple[DataT], List[DataT], Tuple['ArrayT'], List['ArrayT']]
TensorT = Union[ArrayT, Tuple[Tensor], List[Tensor], Tensor, Tuple['TensorT'], List['TensorT']]
TypeT = Union[str]
ShapeT = Union[Tuple[int], List[int]]
IndexT = Union[int, slice, Tuple[Union[int, slice], ...]]

__all__ = [
    'is_tensor',
    'pad_backward',
    'concat_backward',
    'stack_backward',
    'chunk_backward',
    'view_backward',
    'index_select_backward',
    'squeeze_backward',
    'expand_dim_backward',
    'transpose_backward',
    'absolute_backward',
    'around_backward',
    'floor_backward',
    'ceil_backward',
    'clip_backward',
    'negative_backward',
    'log_backward',
    'summation_backward',
    'mean_backward',
    'std_backward',
    'var_backward',
    # 'greater_or_equal_backward',
    # 'greater_backward',
    # 'lesser_or_equal_backward',
    # 'lesser_backward',
    # 'equal_backward',
    # 'not_equal_backward',
    'add_backward',
    'sub_backward',
    'mul_backward',
    'div_backward',
    'power_backward',
    'clone_backward',
    'relu_backward',
    'sigmoid_backward',
    'softmax_backward',
    'tanh_backward',
    'dense_backward',
    'conv_backward',
    'conv_transpose_backward',
    'bilinear_interpolation_backward',
    'dropout_backward',
    'batch_norm_backward',
    'max_pool_backward',
    'avg_pool_backward',
    'lstm_cell_backward',
    'lstm_backward',
    'pad',
    'concat',
    'stack',
    'chunk',
    'view',
    'index_select',
    'zero',
    'one',
    'fill',
    'squeeze',
    'expand_dim',
    'transpose',
    'absolute',
    'around',
    'floor',
    'ceil',
    'clip',
    'negative',
    'log',
    'summation',
    'mean',
    'std',
    'var',
    # 'greater_or_equal',
    # 'greater',
    # 'lesser_or_equal',
    # 'lesser',
    # 'equal',
    # 'not_equal',
    'add',
    'sub',
    'mul',
    'div',
    'power',
    'clone',
    'detach',
    'arange',
    'linspace',
    'normal',
    'uniform',
    'rand',
    'randint',
    'randn',
    'eye',
    'empty',
    'full',
    'zeros',
    'ones',
    'normal_like',
    'uniform_like',
    'rand_like',
    'randint_like',
    'randn_like',
    'eye_like',
    'empty_like',
    'full_like',
    'zeros_like',
    'ones_like',
    'from_array',
    'to_array',
    'half',
    'single',
    'double',
    'cpu',
    'gpu',
    'relu',
    'sigmoid',
    'softmax',
    'tanh',
    'dense',
    'conv',
    'conv_transpose',
    'bilinear_interpolation',
    'dropout',
    'batch_norm',
    'max_pool',
    'avg_pool',
    'lstm_cell',
    'lstm',
    'adam',
    'rmsprop',
]


# need to implement inplace

# should have c / c++ codes to use them in functional apis


def wrapped_partial(func, *args, **kwargs):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def _check_tensor_devices(*tensors: Tensor):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    iterator = iter(tensors)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first.device == x.device for x in iterator)


def _check_tensors(*tensors: Tensor):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    if not _check_tensor_devices(*tensors):
        raise ValueError('devices are not matching')

    if len(tensors) == 0:
        raise ValueError('there should be at least one tensor')

    if tensors[0].device not in ('cpu', 'gpu'):
        raise ValueError("device has to be either 'cpu' or 'gpu'")


def _get_engine(*_: Union[Tensor, str]):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    import numpy as engine

    return engine


def _set_grad(tensor: Tensor, data):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    if not (tensor.requires_grad and not hasattr(tensor, 'retains_grad') and not tensor.is_leaf):
        if not tensor.is_leaf:
            return

        if tensor.grad is None:
            tensor.grad = from_array(data, device=tensor.device)
        else:
            tensor.grad._data = tensor.grad.data + data
    tensor.backward(data)


def _create_tensor(*tensors: Tensor, data, func):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    requires_grad = any(map(lambda x: x.requires_grad, tensors))
    grad_fn = None
    if requires_grad:
        grad_fn = func

    tensor = from_array(data=data, requires_grad=requires_grad, device=tensors[0].device)
    tensor._grad_fn = grad_fn
    if any([x.device == 'gpu' for x in tensors]):
        tensor = tensor.gpu()
    return tensor


def is_tensor(obj: object) -> bool:
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    return isinstance(obj, Tensor)


def pad_backward(gradient: Tensor, inp: Tensor):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    _set_grad(inp, gradient.data * engine.ones_like(inp.data))


def concat_backward(gradient: Tensor, tensors: List[Tensor], axis: int = 0):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(*tensors)
    engine = _get_engine(*tensors)

    grad_arrays = engine.split(gradient.data, len(tensors), axis=axis)
    for idx, tensor in enumerate(tensors):
        _set_grad(tensor, data=grad_arrays[idx] * engine.ones_like(tensor.data))


def stack_backward(gradient: Tensor, tensors: List[Tensor], axis: int = 0):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(*tensors)
    engine = _get_engine(*tensors)

    grad_arrays = engine.split(gradient.data, len(tensors), axis=axis)
    for idx, tensor in enumerate(tensors):
        _set_grad(tensor, data=grad_arrays[idx] * engine.ones_like(tensor.data))


def chunk_backward(gradient: Tensor, tensor: Tensor, chunks: int):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(tensor)
    engine = _get_engine(tensor)

    _set_grad(tensor, gradient.data * engine.ones_like(tensor.data) / chunks)


def view_backward(gradient: Tensor, inp: Tensor):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)

    _set_grad(inp, gradient.data.reshape(inp.shape))


def index_select_backward(gradient: Tensor, inp: Tensor, index: Tensor, dim: int):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    unique, counts = engine.unique(index.data.astype('int'), return_counts=True)
    count_dict = dict(zip(unique, counts))

    index_array = engine.asarray([val for val in range(inp.size(dim))]).astype('int')
    count_array = engine.asarray([count_dict.get(val, 0) for val in range(inp.size(dim))])

    grad_array = engine.zeros_like(gradient.data)
    engine.put_along_axis(grad_array, index_array, count_array, axis=dim)
    _set_grad(inp, data=grad_array)


def squeeze_backward(gradient: Tensor, inp: Tensor):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    _set_grad(inp, gradient.data * engine.ones_like(inp.data))


def expand_dim_backward(gradient: Tensor, inp: Tensor):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    _set_grad(inp, gradient.data * engine.ones_like(inp.data))


def transpose_backward(gradient: Tensor, inp: Tensor):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    _set_grad(inp, gradient.data * engine.ones_like(inp.data))


def absolute_backward(gradient: Tensor, inp: Tensor):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    _set_grad(inp, gradient.data * engine.ones_like(inp.data))


def around_backward(gradient: Tensor, inp: Tensor):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    _set_grad(inp, gradient.data * engine.ones_like(inp.data))


def floor_backward(gradient: Tensor, inp: Tensor):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    _set_grad(inp, gradient.data * engine.ones_like(inp.data))


def ceil_backward(gradient: Tensor, inp: Tensor):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    _set_grad(inp, gradient.data * engine.ones_like(inp.data))


def clip_backward(gradient: Tensor, inp: Tensor):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    _set_grad(inp, gradient.data * engine.ones_like(inp.data))


def negative_backward(gradient: Tensor, inp: Tensor):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    _set_grad(inp, gradient.data * -engine.ones_like(inp.data))


def log_backward(gradient: Tensor, inp: Tensor):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)

    _set_grad(inp, gradient.data * (1 / inp.data))


def summation_backward(gradient: Tensor, inp: Tensor):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    _set_grad(inp, gradient.data * engine.ones_like(inp.data))


def mean_backward(gradient: Tensor, inp: Tensor):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    _set_grad(inp, gradient.data * engine.ones_like(inp.data))


def std_backward(gradient: Tensor, inp: Tensor):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    _set_grad(inp, gradient.data * engine.ones_like(inp.data))


def var_backward(gradient: Tensor, inp: Tensor):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    _set_grad(inp, gradient.data * engine.ones_like(inp.data))


# def greater_or_equal_backward(gradient: Tensor, inp1: Tensor, inp2: Tensor):
#     """
#     Remember the transaction.
#
#     Accepts a state, action, reward, next_state, terminal transaction.
#
#     # Arguments
#         transaction (abstract): state, action, reward, next_state, terminal transaction.
#     """
#     _check_tensors(inp1, inp2)
#     engine = _get_engine(inp1, inp2)
#
#     _set_grad(inp1, gradient.data * engine.ones_like(inp1.data))
#     _set_grad(inp2, gradient.data * engine.ones_like(inp2.data))
#
#
# def greater_backward(gradient: Tensor, inp1: Tensor, inp2: Tensor):
#     """
#     Remember the transaction.
#
#     Accepts a state, action, reward, next_state, terminal transaction.
#
#     # Arguments
#         transaction (abstract): state, action, reward, next_state, terminal transaction.
#     """
#     _check_tensors(inp1, inp2)
#     engine = _get_engine(inp1, inp2)
#
#     _set_grad(inp1, gradient.data * engine.ones_like(inp1.data))
#     _set_grad(inp2, gradient.data * engine.ones_like(inp2.data))
#
#
# def lesser_or_equal_backward(gradient: Tensor, inp1: Tensor, inp2: Tensor):
#     """
#     Remember the transaction.
#
#     Accepts a state, action, reward, next_state, terminal transaction.
#
#     # Arguments
#         transaction (abstract): state, action, reward, next_state, terminal transaction.
#     """
#     _check_tensors(inp1, inp2)
#     engine = _get_engine(inp1, inp2)
#
#     _set_grad(inp1, gradient.data * engine.ones_like(inp1.data))
#     _set_grad(inp2, gradient.data * engine.ones_like(inp2.data))
#
#
# def lesser_backward(gradient: Tensor, inp1: Tensor, inp2: Tensor):
#     """
#     Remember the transaction.
#
#     Accepts a state, action, reward, next_state, terminal transaction.
#
#     # Arguments
#         transaction (abstract): state, action, reward, next_state, terminal transaction.
#     """
#     _check_tensors(inp1, inp2)
#     engine = _get_engine(inp1, inp2)
#
#     _set_grad(inp1, gradient.data * engine.ones_like(inp1.data))
#     _set_grad(inp2, gradient.data * engine.ones_like(inp2.data))
#
#
# def equal_backward(gradient: Tensor, inp1: Tensor, inp2: Tensor):
#     """
#     Remember the transaction.
#
#     Accepts a state, action, reward, next_state, terminal transaction.
#
#     # Arguments
#         transaction (abstract): state, action, reward, next_state, terminal transaction.
#     """
#     _check_tensors(inp1, inp2)
#     engine = _get_engine(inp1, inp2)
#
#     _set_grad(inp1, gradient.data * engine.ones_like(inp1.data))
#     _set_grad(inp2, gradient.data * engine.ones_like(inp2.data))
#
#
# def not_equal_backward(gradient: Tensor, inp1: Tensor, inp2: Tensor):
#     """
#     Remember the transaction.
#
#     Accepts a state, action, reward, next_state, terminal transaction.
#
#     # Arguments
#         transaction (abstract): state, action, reward, next_state, terminal transaction.
#     """
#     _check_tensors(inp1, inp2)
#     engine = _get_engine(inp1, inp2)
#
#     _set_grad(inp1, gradient.data * engine.ones_like(inp1.data))
#     _set_grad(inp2, gradient.data * engine.ones_like(inp2.data))


def add_backward(gradient: Tensor, inp1: Tensor, inp2: Tensor):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp1, inp2)
    engine = _get_engine(inp1, inp2)

    _set_grad(inp1, gradient.data * engine.ones_like(inp1.data))
    _set_grad(inp2, gradient.data * engine.ones_like(inp2.data))


def sub_backward(gradient: Tensor, inp1: Tensor, inp2: Tensor):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp1, inp2)
    engine = _get_engine(inp1, inp2)

    _set_grad(inp1, gradient.data * engine.ones_like(inp1.data))
    _set_grad(inp2, gradient.data * -engine.ones_like(inp2.data))


def mul_backward(gradient: Tensor, inp1: Tensor, inp2: Tensor):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp1, inp2)

    _set_grad(inp1, gradient.data * inp2.data)
    _set_grad(inp2, gradient.data * inp1.data)


def div_backward(gradient: Tensor, inp1: Tensor, inp2: Tensor):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp1, inp2)

    _set_grad(inp1, gradient.data * (1 / inp2.data))
    _set_grad(inp2, gradient.data * inp1.data)


def power_backward(gradient: Tensor, inp: Tensor, p: int):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)

    _set_grad(inp, gradient.data * p * (inp.data ** (p - 1)))


def clone_backward(gradient: Tensor, inp: Tensor):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    _set_grad(inp, gradient.data * engine.ones_like(inp.data))


def relu_backward(gradient: Tensor, inp: Tensor, alpha: float):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    out = engine.where(inp.data > 0, 1, alpha)

    # out[inp.data <= 0] = 0
    # out[inp.data > 0] = 1

    _set_grad(inp, gradient.data * out)


def sigmoid_backward(gradient: Tensor, inp: Tensor, out):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)

    _set_grad(inp, gradient.data * out * (1 - out))


def softmax_backward(gradient: Tensor, inp: Tensor):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    grad_array = gradient.data

    indices = engine.where(grad_array == grad_array.max())

    arr = -grad_array * grad_array
    arr[indices] = grad_array[indices] * (1 - grad_array[indices])

    _set_grad(inp, arr)


def tanh_backward(gradient: Tensor, inp: Tensor, out: Tensor):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    _set_grad(inp, gradient.data * (1 - engine.square(out.data)))


def dense_backward(gradient: Tensor, inp: Tensor, weight: Tensor, bias: Tensor):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp, weight, bias)
    engine = _get_engine(inp, weight, bias)

    _set_grad(inp, engine.dot(gradient.data, weight.data))
    _set_grad(weight, engine.dot(gradient.data.T, inp.data))
    _set_grad(bias, engine.sum(gradient.data, axis=0))


def conv_backward(
        gradient: Tensor, inp: Tensor, weight: Tensor, bias: Tensor, stride: int, padding: Union[List[int], Tuple[int]]
):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp, weight, bias)
    engine = _get_engine(inp, weight, bias)

    _padded_input_array = engine.pad(
        inp.data, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode='constant'
    )
    _weight_array = weight.data
    _grad_array = gradient.data

    _, _, _input_height, input_width = inp.shape
    _, _, _kernel_height, _kernel_width = _weight_array.shape
    _, _, _output_height, _output_width = _grad_array.shape
    _output_array = engine.zeros_like(_padded_input_array)

    _weight_grad = engine.zeros_like(_weight_array)
    _bias_grad = _grad_array.sum(axis=(0, 2, 3))

    for _row in range(_output_height):
        for _column in range(_output_width):
            _output_array[:, :, _row * stride: _row * stride + _kernel_height, _column * stride: _column * stride + _kernel_width, ] += engine.sum(
                _weight_array[None, :, :, :, :] * _grad_array[:, :, None, _row: _row + 1, _column: _column + 1],
                axis=1,
            )
            _weight_grad += engine.sum(
                _padded_input_array[:, None, :, _row * stride: _row * stride + _kernel_height, _column * stride: _column * stride + _kernel_width, ]
                * _grad_array[:, :, None, _row: _row + 1, _column: _column + 1],
                axis=0,
            )

    _set_grad(inp, _weight_grad)
    _set_grad(weight, _bias_grad)
    _set_grad(bias, _output_array[:, :, padding[0]: padding[0] + _input_height, padding[1]: padding[1] + input_width])


def conv_transpose_backward(
        gradient: Tensor, inp: Tensor, weight: Tensor, bias: Tensor, stride: int, padding: Union[List[int], Tuple[int]]
):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    # _check_tensors(inp, weight, bias)
    # engine = _get_engine(inp, weight, bias)
    #
    # _padded_input_array = engine.pad(
    #     inp.data, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode='constant'
    # )
    # _weight_array = weight.data
    # _grad_array = gradient.data
    #
    # _, _, _input_height, input_width = inp.shape
    # _, _, _kernel_height, _kernel_width = _weight_array.shape
    # _, _, _output_height, _output_width = _grad_array.shape
    # _output_array = engine.zeros_like(_padded_input_array)
    #
    # _weight_grad = engine.zeros_like(_weight_array)
    # _bias_grad = _grad_array.sum(axis=(0, 2, 3))
    #
    # for _row in range(_output_height):
    #     for _column in range(_output_width):
    #         _output_array[:, :, _row * stride: _row * stride + _kernel_height, _column * stride: _column * stride + _kernel_width, ] += engine.sum(
    #             _weight_array[None, :, :, :, :] * _grad_array[:, :, None, _row: _row + 1, _column: _column + 1],
    #             axis=1,
    #         )
    #         _weight_grad += engine.sum(
    #             _padded_input_array[:, None, :, _row * stride: _row * stride + _kernel_height, _column * stride: _column * stride + _kernel_width, ]
    #             * _grad_array[:, :, None, _row: _row + 1, _column: _column + 1],
    #             axis=0,
    #         )
    #
    # _set_grad(inp, _weight_grad)
    # _set_grad(weight, _bias_grad)
    # _set_grad(bias, _output_array[:, :, padding[0]: padding[0] + _input_height, padding[1]: padding[1] + input_width])


def bilinear_interpolation_backward(gradient: Tensor, inp: Tensor, scale_factor):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    b, c, height, width = inp.data.shape

    if isinstance(scale_factor, (list, tuple)):  # make sure they have same dimensions
        scale_factors = scale_factor
    else:
        scale_factors = [scale_factor for _ in range(2)]

    output_size = [int(inp.data.shape[idx + 2] * scale_factors[idx]) for idx in range(2)]

    # Initialize the output image
    interpolated_image = engine.zeros([b, c] + output_size, dtype=engine.float32)

    new_grad = engine.zeros_like(inp.data)

    for batch in range(b):
        for y in range(output_size[0]):
            for x in range(output_size[1]):
                # Calculate the corresponding coordinates in the original image
                original_x = x / scale_factors[1]
                original_y = y / scale_factors[0]

                # Find the four nearest neighbor pixels in the original image
                x0 = int(original_x)
                x1 = min(x0 + 1, width - 1)
                y0 = int(original_y)
                y1 = min(y0 + 1, height - 1)

                # Calculate the weights for bilinear interpolation
                weight_x1 = original_x - x0
                weight_x0 = 1 - weight_x1
                weight_y1 = original_y - y0
                weight_y0 = 1 - weight_y1

                # Perform bilinear interpolation for each color channel
                for channel in range(c):
                    interpolated_value = (
                            inp.data[batch, channel, y0, x0] * weight_x0 * weight_y0 +
                            inp.data[batch, channel, y0, x1] * weight_x1 * weight_y0 +
                            inp.data[batch, channel, y1, x0] * weight_x0 * weight_y1 +
                            inp.data[batch, channel, y1, x1] * weight_x1 * weight_y1
                    )
                    interpolated_image[batch, channel, y, x] = interpolated_value

                    new_grad[batch, channel, y0, x0] += weight_x0 * weight_y0 * gradient.data[batch, channel, y, x]
                    new_grad[batch, channel, y0, x1] += weight_x1 * weight_y0 * gradient.data[batch, channel, y, x]
                    new_grad[batch, channel, y1, x0] += weight_x0 * weight_y1 * gradient.data[batch, channel, y, x]
                    new_grad[batch, channel, y1, x1] += weight_x1 * weight_y1 * gradient.data[batch, channel, y, x]

    _set_grad(inp, new_grad)


def dropout_backward(gradient: Tensor, inp: Tensor, mask, keep_prob: float):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    def apply_mask(array) -> engine.array:
        array *= mask
        array /= keep_prob
        return array

    _set_grad(inp, apply_mask(gradient))


def batch_norm_backward(gradient: Tensor, inp: Tensor, weight: Tensor, bias: Tensor, training: bool, **kwargs):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    if training:
        batch_size = inp.data.shape[0]
        weight_by_grad = weight.data * gradient.data
        dxc = weight_by_grad / kwargs['input_standard_deviation']
        dstd = -engine.sum(
            (weight_by_grad * kwargs['input_mean_difference'])
            / (kwargs['input_standard_deviation'] * kwargs['input_standard_deviation']),
            axis=0,
        )
        dvar = 0.5 * dstd / kwargs['input_standard_deviation']
        dxc += (2.0 / batch_size) * kwargs['input_mean_difference'] * dvar
        dmu = engine.sum(dxc, axis=0)

        _set_grad(inp, dxc - dmu / batch_size)
        _set_grad(weight, engine.sum(kwargs['input_mean_over_input_standard_deviation'] * gradient.data, axis=0))
        _set_grad(bias, gradient.data.sum(axis=0))
    else:
        weight_by_grad = weight.data * gradient.data

        _set_grad(inp, weight_by_grad / kwargs['input_standard_deviation'])
        _set_grad(weight, engine.sum(kwargs['input_mean_over_input_standard_deviation'] * gradient.data, axis=0))
        _set_grad(bias, gradient.data.sum(axis=0))


def max_pool_backward(
        gradient: Tensor,
        inp: Tensor,
        kernel_size: Union[List[int], Tuple[int]],
        stride: int,
        padding: Union[List[int], Tuple[int]],
        cache: dict,
):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    grad_array = gradient.data

    _, _, _output_height, _output_width = grad_array.shape
    _kernel_height, _kernel_width = kernel_size

    _padded_input_array = engine.pad(
        inp.data, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode='constant'
    )
    _output_array = engine.zeros_like(_padded_input_array)

    for _row in range(_output_height):
        for _column in range(_output_width):
            increment = grad_array[:, :, _row: _row + 1, _column: _column + 1] * cache[(_row, _column)]
            _output_array[:, :, _row * stride: _row * stride + _kernel_height, _column * stride: _column * stride + _kernel_width, ] += increment

    _set_grad(
        inp,
        _output_array[:, :, padding[0]: padding[0] + _output_height - 1, padding[1]: padding[1] + _output_width - 1],
    )


def avg_pool_backward(
        gradient: Tensor,
        inp: Tensor,
        kernel_size: Union[List[int], Tuple[int]],
        stride: int,
        padding: Union[List[int], Tuple[int]],
):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    grad_array = gradient.data

    _, _, _output_height, _output_width = grad_array.shape
    _kernel_height, _kernel_width = kernel_size

    _padded_input_array = engine.pad(
        inp.data, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode='constant'
    )
    _output_array = engine.zeros_like(_padded_input_array)

    for _row in range(_output_height):
        for _column in range(_output_width):
            increment = grad_array[:, :, _row: _row + 1, _column: _column + 1] / _kernel_height / _kernel_width
            _output_array[:, :, _row * stride: _row * stride + _kernel_height, _column * stride: _column * stride + _kernel_width, ] += increment

    _set_grad(
        inp,
        _output_array[:, :, padding[0]: padding[0] + _output_height - 1, padding[1]: padding[1] + _output_width - 1],
    )


def lstm_cell_backward(gradient, inp, all_weights, cache):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    engine = _get_engine()

    gradient_array = gradient.data.copy()
    inp_array = inp.data.copy()

    out = cache['out']
    hx = cache['hx']
    cx = cache['cx']
    igates = cache['i']
    fgates = cache['f']
    cgates = cache['c']
    ogates = cache['o']

    batch_size, layer_num, time_sequence, hidden_size = hx.shape

    w_ih_arrays = [x[0].data for x in all_weights]
    w_hh_arrays = [x[1].data for x in all_weights]

    delta_array = engine.zeros((batch_size, layer_num, time_sequence, hidden_size))
    delta_out_array = engine.zeros((batch_size, layer_num, time_sequence, hidden_size))
    d_out_array = engine.zeros((batch_size, layer_num, time_sequence, hidden_size))
    d_state_array = engine.zeros((batch_size, layer_num, time_sequence, hidden_size))
    d_cgate_array = engine.zeros((batch_size, layer_num, time_sequence, hidden_size))
    d_igate_array = engine.zeros((batch_size, layer_num, time_sequence, hidden_size))
    d_fgate_array = engine.zeros((batch_size, layer_num, time_sequence, hidden_size))
    d_ogate_array = engine.zeros((batch_size, layer_num, time_sequence, hidden_size))

    d_inp_array = engine.zeros_like(inp_array)

    d_wih_array = [engine.zeros_like(x[0].data) for x in all_weights]
    d_whh_array = [engine.zeros_like(x[1].data) for x in all_weights]
    d_bih_array = [engine.zeros_like(x[2].data) for x in all_weights]
    d_bhh_array = [engine.zeros_like(x[3].data) for x in all_weights]

    for time in range(time_sequence - 1, -1, -1):
        a = 0
        for layer in range(layer_num - 1, -1, -1):
            w_ih_array = w_ih_arrays[layer]
            w_hh_array = w_hh_arrays[layer]

            if time == time_sequence - 1:
                if layer == layer_num - 1:
                    cell_grad = gradient_array[:, time, :]
                else:
                    cell_grad = a
            else:
                if layer == layer_num - 1:
                    cell_grad = gradient_array[:, time, :]
                else:
                    cell_grad = a

            delta_array[:, layer, time, :] = cell_grad
            if time == time_sequence - 1:
                delta_out = engine.zeros_like(delta_out_array[:, layer, time, :])
            else:
                delta_out = delta_out_array[:, layer, time + 1, :]

            d_out_array[:, layer, time, :] = delta_array[:, layer, time, :] + delta_out

            if time == time_sequence - 1:
                d_next_state = engine.zeros_like(d_state_array[:, layer, time, :])
            else:
                d_next_state = d_state_array[:, layer, time + 1, :]

            if time == time_sequence - 1:
                next_forget = engine.zeros_like(fgates[:, layer, time, :])
            else:
                next_forget = fgates[:, layer, time + 1, :]

            d_state_array[:, layer, time, :] = (
                    d_out_array[:, layer, time, :]
                    * ogates[:, layer, time, :]
                    * (1 - engine.tanh(cx[:, layer, time, :]) ** 2)
                    + d_next_state * next_forget
            )

            if time == 0:
                prev_state = engine.zeros_like(cx[:, layer, time, :])
            else:
                prev_state = cx[:, layer, time - 1, :]

            d_cgate = (
                    d_state_array[:, layer, time, :] * igates[:, layer, time, :] * (1 - cgates[:, layer, time, :] ** 2)
            )
            d_igate = (
                    d_state_array[:, layer, time, :]
                    * cgates[:, layer, time, :]
                    * igates[:, layer, time, :]
                    * (1 - igates[:, layer, time, :])
            )
            d_fgate = (
                    d_state_array[:, layer, time, :]
                    * prev_state
                    * fgates[:, layer, time, :]
                    * (1 - fgates[:, layer, time, :])
            )
            d_ogate = (
                    d_out_array[:, layer, time, :]
                    * engine.tanh(cx[:, layer, time, :])
                    * ogates[:, layer, time, :]
                    * (1 - ogates[:, layer, time, :])
            )

            d_cgate_array[:, layer, time, :] = d_cgate
            d_igate_array[:, layer, time, :] = d_igate
            d_fgate_array[:, layer, time, :] = d_fgate
            d_ogate_array[:, layer, time, :] = d_ogate

            d_gates = engine.hstack([d_igate, d_fgate, d_cgate, d_ogate])
            # d_x_array[:, layer, time, :] = (w_ih_array.T @ d_gates.T).T
            a = (w_ih_array.T @ d_gates.T).T

            delta_out_array[:, layer, time, :] = (w_hh_array.T @ d_gates.T).T

            # d_wih = engine.outer(d_gates, inp[:, time, :])
            if layer == 0:
                i = inp_array[:, time, :]
            else:
                i = out[:, layer - 1, time, :]
            # d_wih = d_gates.T @ inp_array[:, time, :]
            d_wih = d_gates.T @ i

            if time == time_sequence - 1:
                d_gates_next = engine.hstack(
                    [
                        engine.zeros_like(d_igate_array[:, layer, time, :]),
                        engine.zeros_like(d_fgate_array[:, layer, time, :]),
                        engine.zeros_like(d_cgate_array[:, layer, time, :]),
                        engine.zeros_like(d_ogate_array[:, layer, time, :]),
                    ]
                )
            else:
                d_gates_next = engine.hstack(
                    [
                        d_igate_array[:, layer, time + 1, :],
                        d_fgate_array[:, layer, time + 1, :],
                        d_cgate_array[:, layer, time + 1, :],
                        d_ogate_array[:, layer, time + 1, :],
                    ]
                )
            d_whh = d_gates_next.T @ out[:, layer, time, :]

            d_wih_array[layer] += d_wih
            d_whh_array[layer] += d_whh
            d_bih_array[layer] += engine.sum(d_gates.T, axis=1)
            d_bhh_array[layer] += engine.sum(d_gates.T, axis=1)

        d_inp_array[:, time, :] = a

    _set_grad(inp, d_inp_array)
    for layer in range(layer_num):
        _set_grad(all_weights[layer][0], d_wih_array[layer])
        _set_grad(all_weights[layer][1], d_whh_array[layer])
        _set_grad(all_weights[layer][2], d_bih_array[layer])
        _set_grad(all_weights[layer][3], d_bhh_array[layer])


def lstm_backward(gradient, inp, all_weights, cache):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    engine = _get_engine()

    gradient_array = gradient.data.copy()
    inp_array = inp.data.copy()

    out = cache['out']
    hx = cache['hx']
    cx = cache['cx']
    igates = cache['i']
    fgates = cache['f']
    cgates = cache['c']
    ogates = cache['o']

    batch_size, layer_num, time_sequence, hidden_size = hx.shape

    w_ih_arrays = [x[0].data for x in all_weights]
    w_hh_arrays = [x[1].data for x in all_weights]

    delta_array = engine.zeros((batch_size, layer_num, time_sequence, hidden_size))
    delta_out_array = engine.zeros((batch_size, layer_num, time_sequence, hidden_size))
    d_out_array = engine.zeros((batch_size, layer_num, time_sequence, hidden_size))
    d_state_array = engine.zeros((batch_size, layer_num, time_sequence, hidden_size))
    d_cgate_array = engine.zeros((batch_size, layer_num, time_sequence, hidden_size))
    d_igate_array = engine.zeros((batch_size, layer_num, time_sequence, hidden_size))
    d_fgate_array = engine.zeros((batch_size, layer_num, time_sequence, hidden_size))
    d_ogate_array = engine.zeros((batch_size, layer_num, time_sequence, hidden_size))

    d_inp_array = engine.zeros_like(inp_array)

    d_wih_array = [engine.zeros_like(x[0].data) for x in all_weights]
    d_whh_array = [engine.zeros_like(x[1].data) for x in all_weights]
    d_bih_array = [engine.zeros_like(x[2].data) for x in all_weights]
    d_bhh_array = [engine.zeros_like(x[3].data) for x in all_weights]

    for time in range(time_sequence - 1, -1, -1):
        a = 0
        for layer in range(layer_num - 1, -1, -1):
            w_ih_array = w_ih_arrays[layer]
            w_hh_array = w_hh_arrays[layer]

            if time == time_sequence - 1:
                if layer == layer_num - 1:
                    cell_grad = gradient_array[:, time, :]
                else:
                    cell_grad = a
            else:
                if layer == layer_num - 1:
                    cell_grad = gradient_array[:, time, :]
                else:
                    cell_grad = a

            delta_array[:, layer, time, :] = cell_grad
            if time == time_sequence - 1:
                delta_out = engine.zeros_like(delta_out_array[:, layer, time, :])
            else:
                delta_out = delta_out_array[:, layer, time + 1, :]

            d_out_array[:, layer, time, :] = delta_array[:, layer, time, :] + delta_out

            if time == time_sequence - 1:
                d_next_state = engine.zeros_like(d_state_array[:, layer, time, :])
            else:
                d_next_state = d_state_array[:, layer, time + 1, :]

            if time == time_sequence - 1:
                next_forget = engine.zeros_like(fgates[:, layer, time, :])
            else:
                next_forget = fgates[:, layer, time + 1, :]

            d_state_array[:, layer, time, :] = (
                    d_out_array[:, layer, time, :]
                    * ogates[:, layer, time, :]
                    * (1 - engine.tanh(cx[:, layer, time, :]) ** 2)
                    + d_next_state * next_forget
            )

            if time == 0:
                prev_state = engine.zeros_like(cx[:, layer, time, :])
            else:
                prev_state = cx[:, layer, time - 1, :]

            d_cgate = (
                    d_state_array[:, layer, time, :] * igates[:, layer, time, :] * (1 - cgates[:, layer, time, :] ** 2)
            )
            d_igate = (
                    d_state_array[:, layer, time, :]
                    * cgates[:, layer, time, :]
                    * igates[:, layer, time, :]
                    * (1 - igates[:, layer, time, :])
            )
            d_fgate = (
                    d_state_array[:, layer, time, :]
                    * prev_state
                    * fgates[:, layer, time, :]
                    * (1 - fgates[:, layer, time, :])
            )
            d_ogate = (
                    d_out_array[:, layer, time, :]
                    * engine.tanh(cx[:, layer, time, :])
                    * ogates[:, layer, time, :]
                    * (1 - ogates[:, layer, time, :])
            )

            d_cgate_array[:, layer, time, :] = d_cgate
            d_igate_array[:, layer, time, :] = d_igate
            d_fgate_array[:, layer, time, :] = d_fgate
            d_ogate_array[:, layer, time, :] = d_ogate

            d_gates = engine.hstack([d_igate, d_fgate, d_cgate, d_ogate])
            # d_x_array[:, layer, time, :] = (w_ih_array.T @ d_gates.T).T
            a = (w_ih_array.T @ d_gates.T).T

            delta_out_array[:, layer, time, :] = (w_hh_array.T @ d_gates.T).T

            # d_wih = engine.outer(d_gates, inp[:, time, :])
            if layer == 0:
                i = inp_array[:, time, :]
            else:
                i = out[:, layer - 1, time, :]
            # d_wih = d_gates.T @ inp_array[:, time, :]
            d_wih = d_gates.T @ i

            if time == time_sequence - 1:
                d_gates_next = engine.hstack(
                    [
                        engine.zeros_like(d_igate_array[:, layer, time, :]),
                        engine.zeros_like(d_fgate_array[:, layer, time, :]),
                        engine.zeros_like(d_cgate_array[:, layer, time, :]),
                        engine.zeros_like(d_ogate_array[:, layer, time, :]),
                    ]
                )
            else:
                d_gates_next = engine.hstack(
                    [
                        d_igate_array[:, layer, time + 1, :],
                        d_fgate_array[:, layer, time + 1, :],
                        d_cgate_array[:, layer, time + 1, :],
                        d_ogate_array[:, layer, time + 1, :],
                    ]
                )
            d_whh = d_gates_next.T @ out[:, layer, time, :]

            d_wih_array[layer] += d_wih
            d_whh_array[layer] += d_whh
            d_bih_array[layer] += engine.sum(d_gates.T, axis=1)
            d_bhh_array[layer] += engine.sum(d_gates.T, axis=1)

        d_inp_array[:, time, :] = a

    _set_grad(inp, d_inp_array)
    for layer in range(layer_num):
        _set_grad(all_weights[layer][0], d_wih_array[layer])
        _set_grad(all_weights[layer][1], d_whh_array[layer])
        _set_grad(all_weights[layer][2], d_bih_array[layer])
        _set_grad(all_weights[layer][3], d_bhh_array[layer])


def pad(inp: Tensor, padding, mode="constant") -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    left, right, top, bottom = padding

    return _create_tensor(
        inp, data=engine.pad(
            inp.data, ((0, 0), (0, 0), (top, bottom), (left, right)), mode=mode
        ), func=wrapped_partial(pad_backward, inp=inp)
    )


def concat(tensors: List[Tensor], axis: int = 0) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(*tensors)
    engine = _get_engine(*tensors)

    return _create_tensor(
        *tensors,
        data=engine.concatenate(list(map(lambda x: x.data, tensors)), axis=axis),
        func=wrapped_partial(concat_backward, tensors=tensors, axis=axis)
    )


def stack(tensors: List[Tensor], axis: int = 0) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(*tensors)
    engine = _get_engine(*tensors)

    return _create_tensor(
        *tensors,
        data=engine.stack(list(map(lambda x: x.data, tensors)), axis=axis),
        func=wrapped_partial(stack_backward, tensors=tensors, axis=axis)
    )


def chunk(tensor: Tensor, chunks: int, dim: int = 0):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(tensor)
    engine = _get_engine(tensor)

    arrays = engine.split(tensor.data, chunks, dim)

    tensors = []
    for array in arrays:
        tensors.append(
            _create_tensor(tensor, data=array, func=wrapped_partial(chunk_backward, tensor=tensor, chunks=chunks))
        )
    return tensors


def view(inp, size=None) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)

    return _create_tensor(inp, data=inp.data.reshape(size), func=wrapped_partial(view_backward, inp))


def index_select(inp, dim, index) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    return _create_tensor(
        inp,
        data=engine.take_along_axis(inp.data, index.data.astype('int'), dim),
        func=wrapped_partial(index_select_backward, inp=inp, index=index, dim=dim),
    )


def zero(inp) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    inp._data = engine.zeros_like(inp.data)
    return inp


def one(inp) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    inp._data = engine.ones_like(inp.data)
    return inp


def fill(inp, value) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)

    inp.data.fill(value)
    return inp


def squeeze(inp, axis=None) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    return _create_tensor(
        inp, data=engine.squeeze(inp.data, axis=axis), func=wrapped_partial(squeeze_backward, inp=inp)
    )


def expand_dim(inp, axis=None) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    return _create_tensor(
        inp, data=engine.expand_dims(inp.data, axis=axis), func=wrapped_partial(expand_dim_backward, inp=inp)
    )


def transpose(inp, axes=None) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    return _create_tensor(
        inp, data=engine.transpose(inp.data, axes=axes), func=wrapped_partial(transpose_backward, inp=inp)
    )


def absolute(inp) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    return _create_tensor(inp, data=engine.absolute(inp.data), func=wrapped_partial(absolute_backward, inp=inp))


def around(inp) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    return _create_tensor(inp, data=engine.around(inp.data), func=wrapped_partial(around_backward, inp=inp))


def floor(inp) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    return _create_tensor(inp, data=engine.floor(inp.data), func=wrapped_partial(floor_backward, inp=inp))


def ceil(inp) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    return _create_tensor(inp, data=engine.ceil(inp.data), func=wrapped_partial(ceil_backward, inp=inp))


def clip(inp, min_val, max_val) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    return _create_tensor(
        inp, data=engine.clip(inp.data, min_val, max_val), func=wrapped_partial(clip_backward, inp=inp)
    )


def negative(inp) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    return _create_tensor(inp, data=engine.negative(inp.data), func=wrapped_partial(negative_backward, inp=inp))


def log(inp) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    # print(inp.data)
    return _create_tensor(inp, data=engine.log(inp.data), func=wrapped_partial(log_backward, inp=inp))


def summation(inp) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    return _create_tensor(inp, data=engine.sum(inp.data), func=wrapped_partial(summation_backward, inp=inp))


def mean(inp) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    return _create_tensor(inp, data=engine.mean(inp.data), func=wrapped_partial(mean_backward, inp=inp))


def std(inp) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    return _create_tensor(inp, data=engine.std(inp.data), func=wrapped_partial(std_backward, inp=inp))


def var(inp) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    return _create_tensor(inp, data=engine.var(inp.data), func=wrapped_partial(var_backward, inp=inp))


# def greater_or_equal(inp1, inp2) -> 'Tensor':
#     """
#     Remember the transaction.
#
#     Accepts a state, action, reward, next_state, terminal transaction.
#
#     # Arguments
#         transaction (abstract): state, action, reward, next_state, terminal transaction.
#     """
#     if not isinstance(inp2, Tensor):
#         inp2 = from_array(inp2, device=inp1.device)
#
#     _check_tensors(inp1, inp2)
#
#     return _create_tensor(
#         inp1, inp2, data=inp1.data >= inp2.data, func=wrapped_partial(greater_or_equal_backward, inp1=inp1, inp2=inp2)
#     )
#
#
# def greater(inp1, inp2) -> 'Tensor':
#     """
#     Remember the transaction.
#
#     Accepts a state, action, reward, next_state, terminal transaction.
#
#     # Arguments
#         transaction (abstract): state, action, reward, next_state, terminal transaction.
#     """
#     if not isinstance(inp2, Tensor):
#         inp2 = from_array(inp2, device=inp1.device)
#
#     _check_tensors(inp1, inp2)
#
#     return _create_tensor(
#         inp1, inp2, data=inp1.data > inp2.data, func=wrapped_partial(greater_backward, inp1=inp1, inp2=inp2)
#     )
#
#
# def lesser_or_equal(inp1, inp2) -> 'Tensor':
#     """
#     Remember the transaction.
#
#     Accepts a state, action, reward, next_state, terminal transaction.
#
#     # Arguments
#         transaction (abstract): state, action, reward, next_state, terminal transaction.
#     """
#     if not isinstance(inp2, Tensor):
#         inp2 = from_array(inp2, device=inp1.device)
#
#     _check_tensors(inp1, inp2)
#
#     return _create_tensor(
#         inp1, inp2, data=inp1.data <= inp2.data, func=wrapped_partial(lesser_or_equal_backward, inp1=inp1, inp2=inp2)
#     )
#
#
# def lesser(inp1, inp2) -> 'Tensor':
#     """
#     Remember the transaction.
#
#     Accepts a state, action, reward, next_state, terminal transaction.
#
#     # Arguments
#         transaction (abstract): state, action, reward, next_state, terminal transaction.
#     """
#     if not isinstance(inp2, Tensor):
#         inp2 = from_array(inp2, device=inp1.device)
#
#     _check_tensors(inp1, inp2)
#
#     return _create_tensor(
#         inp1, inp2, data=inp1.data < inp2.data, func=wrapped_partial(lesser_backward, inp1=inp1, inp2=inp2)
#     )
#
#
# def equal(inp1, inp2) -> 'Tensor':
#     """
#     Remember the transaction.
#
#     Accepts a state, action, reward, next_state, terminal transaction.
#
#     # Arguments
#         transaction (abstract): state, action, reward, next_state, terminal transaction.
#     """
#     if not isinstance(inp2, Tensor):
#         inp2 = from_array(inp2, device=inp1.device)
#
#     _check_tensors(inp1, inp2)
#
#     return _create_tensor(
#         inp1, inp2, data=inp1.data == inp2.data, func=wrapped_partial(equal_backward, inp1=inp1, inp2=inp2)
#     )
#
#
# def not_equal(inp1, inp2) -> 'Tensor':
#     """
#     Remember the transaction.
#
#     Accepts a state, action, reward, next_state, terminal transaction.
#
#     # Arguments
#         transaction (abstract): state, action, reward, next_state, terminal transaction.
#     """
#     if not isinstance(inp2, Tensor):
#         inp2 = from_array(inp2, device=inp1.device)
#
#     _check_tensors(inp1, inp2)
#
#     return _create_tensor(
#         inp1, inp2, data=inp1.data != inp2.data, func=wrapped_partial(not_equal_backward, inp1=inp1, inp2=inp2)
#     )


def add(inp1, inp2) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    if not isinstance(inp2, Tensor):
        inp2 = from_array(inp2, device=inp1.device)

    _check_tensors(inp1, inp2)

    return _create_tensor(
        inp1, inp2, data=inp1.data + inp2.data, func=wrapped_partial(add_backward, inp1=inp1, inp2=inp2)
    )


def sub(inp1, inp2) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    if not isinstance(inp2, Tensor):
        inp2 = from_array(inp2, device=inp1.device)

    _check_tensors(inp1, inp2)

    return _create_tensor(
        inp1, inp2, data=inp1.data - inp2.data, func=wrapped_partial(sub_backward, inp1=inp1, inp2=inp2)
    )


def mul(inp1, inp2) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    if not isinstance(inp2, Tensor):
        inp2 = from_array(inp2, device=inp1.device)

    _check_tensors(inp1, inp2)

    return _create_tensor(
        inp1, inp2, data=inp1.data * inp2.data, func=wrapped_partial(mul_backward, inp1=inp1, inp2=inp2)
    )


def div(inp1, inp2) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    if not isinstance(inp2, Tensor):
        inp2 = from_array(inp2, device=inp1.device)

    _check_tensors(inp1, inp2)

    return _create_tensor(
        inp1, inp2, data=inp1.data / inp2.data, func=wrapped_partial(div_backward, inp1=inp1, inp2=inp2)
    )


def power(inp, p) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    # if not isinstance(p, Tensor):
    #     p = from_array(p, device=inp.device)

    _check_tensors(inp)

    return _create_tensor(inp, data=inp.data ** p, func=wrapped_partial(power_backward, inp=inp, p=p))


def clone(inp) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)

    return _create_tensor(inp, data=inp.data, func=wrapped_partial(clone_backward, inp=inp))


def detach(inp, inplace=True) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)

    if inplace:
        inp._grad_fn = None
        inp._requires_grad = False
        return inp

    _clone = inp.clone()
    _clone._grad_fn = None
    _clone._requires_grad = False
    return _clone


def arange(start=0, stop=0, step=1, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    engine = _get_engine(device)
    return from_array(engine.arange(start, stop, step).astype(dtype), requires_grad, device)


def linspace(start, end, steps, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    engine = _get_engine(device)
    return from_array(engine.linspace(start, end, steps).astype(dtype), requires_grad, device)


def normal(loc=0.0, scale=1.0, size=None, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    engine = _get_engine(device)
    return from_array(engine.random.normal(loc, scale, size).astype(dtype), requires_grad, device)


def uniform(low=-1.0, high=1.0, size=None, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    engine = _get_engine(device)
    return from_array(engine.random.uniform(low, high, size).astype(dtype), requires_grad, device)


def rand(size, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    engine = _get_engine(device)
    return from_array(engine.random.rand(size).astype(dtype), requires_grad, device)


def randint(low=0, high=0, size=None, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    engine = _get_engine(device)
    return from_array(engine.random.randint(low, high, *size).astype(dtype), requires_grad, device)


def randn(size, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    engine = _get_engine(device)
    return from_array(engine.random.randn(size).astype(dtype), requires_grad, device)


def eye(rows, columns=None, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    engine = _get_engine(device)
    return from_array(engine.eye(rows, columns).astype(dtype), requires_grad, device)


def empty(size, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    engine = _get_engine(device)
    return from_array(engine.zeros(size).astype(dtype), requires_grad, device)


def full(size, fill_value, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    engine = _get_engine(device)
    return from_array(engine.full(size, fill_value).astype(dtype), requires_grad, device)


def zeros(size, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    engine = _get_engine(device)
    return from_array(engine.zeros(size).astype(dtype), requires_grad, device)


def ones(size, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    engine = _get_engine(device)
    return from_array(engine.ones(size).astype(dtype), requires_grad, device)


def normal_like(tensor, loc=0.0, scale=1.0, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    return normal(loc, scale, tensor.shape, requires_grad, device, dtype)


def uniform_like(tensor, low=-1.0, high=1.0, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    return uniform(low, high, tensor.shape, requires_grad, device, dtype)


def rand_like(tensor, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    return rand(tensor.shape, requires_grad, device, dtype)


def randint_like(tensor, low=0, high=0, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    return randint(low, high, tensor.shape, requires_grad, device, dtype)


def randn_like(tensor, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    return randn(tensor.shape, requires_grad, device, dtype)


def eye_like(tensor, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    return eye(tensor.shape[0], tensor.shape[1], requires_grad, device, dtype)


def empty_like(tensor, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    return empty(tensor.shape, requires_grad, device, dtype)


def full_like(tensor, fill_value, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    return full(tensor.shape, fill_value, requires_grad, device, dtype)


def zeros_like(tensor, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    return zeros(tensor.shape, requires_grad, device, dtype)


def ones_like(tensor, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    return ones(tensor.shape, requires_grad, device, dtype)


def from_array(data, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    engine = _get_engine('cpu')
    tensor = Tensor(engine.copy(data).astype(dtype), requires_grad=requires_grad)
    if device == 'gpu':
        tensor = tensor.gpu()
    return tensor


def to_array(inp):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine('cpu')

    if inp.device != 'cpu':
        raise TypeError(
            "can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first."
        )
    if inp.requires_grad:
        raise RuntimeError("Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.")
    return engine.copy(inp.data)


def half(inp) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    inp._data = inp.data.astype('float16')
    return inp


def single(inp) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    inp._data = inp.data.astype('float32')
    return inp


def double(inp) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    inp._data = inp.data.astype('float64')
    return inp


def cpu(inp) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    if inp.device == 'cpu':
        return inp

    inp._device = 'cpu'
    inp._data = _get_engine().array(inp.data)
    # inp._data = cp.asnumpy(inp.data)
    return inp


def gpu(inp) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    if inp.device == 'gpu':
        return inp

    inp._device = 'gpu'
    inp._data = _get_engine().array(inp.data)
    # inp._data = cp.array(inp.data)
    return inp


def relu(inp, alpha) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)

    arr = inp.data
    # arr[arr <= 0] = 0
    arr = _get_engine().where(arr > 0, arr, arr * alpha)
    return _create_tensor(inp, data=arr, func=wrapped_partial(relu_backward, inp=inp, alpha=alpha))


def sigmoid(inp) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    output_array = 1 / (1 + engine.exp(-inp.data))
    return _create_tensor(inp, data=output_array, func=wrapped_partial(sigmoid_backward, inp=inp, out=output_array))


def softmax(inp) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    e = engine.exp(inp.data - inp.data.max(axis=1, keepdims=True))
    z = e / engine.sum(e, axis=1, keepdims=True)
    return _create_tensor(inp, data=z, func=wrapped_partial(softmax_backward, inp=inp))


def tanh(inp) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    output_array = engine.tanh(inp.data)
    return _create_tensor(inp, data=output_array, func=wrapped_partial(tanh_backward, inp=inp, out=output_array))


def dense(inp, weight, bias) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp, weight, bias)
    engine = _get_engine(inp, weight, bias)

    return _create_tensor(
        inp,
        weight,
        bias,
        data=engine.dot(inp.data, weight.data.T) + bias.data,
        func=wrapped_partial(dense_backward, inp=inp, weight=weight, bias=bias),
    )


def conv(inp, weight, bias, stride, padding) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp, weight, bias)
    engine = _get_engine(inp, weight, bias)

    padded_input_array = engine.pad(
        inp.data, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode='constant'
    )
    weight_array = weight.data
    bias_array = bias.data

    output_shape = _calculate_output_dims(
        input_shape=inp.shape, kernel_shape=weight.shape, padding=padding, stride=stride
    )

    output_array = engine.zeros(output_shape)

    _, _, kernel_height, kernel_width = weight.shape
    _, _, output_height, output_width = output_shape

    for row in range(output_height):
        for column in range(output_width):
            output_array[:, :, row, column] = engine.sum(
                padded_input_array[:, None, :, row * stride: row * stride + kernel_height, column * stride: column * stride + kernel_width, ]
                * weight_array[None, :, :, :],
                axis=(2, 3, 4),
            )

    return _create_tensor(
        inp,
        weight,
        bias,
        data=output_array + bias_array[:, None, None],
        func=wrapped_partial(conv_backward, inp=inp, weight=weight, bias=bias, stride=stride, padding=padding),
    )


def conv_transpose(inp, weight, bias, stride, padding) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp, weight, bias)
    engine = _get_engine(inp, weight, bias)

    weight_array = weight.data
    bias_array = bias.data

    input_shape = inp.data.shape
    output_shape = _calculate_input_dims(
        output_shape=input_shape, kernel_shape=weight.shape, padding=padding, stride=stride
    )

    output_array = engine.zeros(output_shape)

    _, _, kernel_height, kernel_width = weight.shape
    _, _, input_height, input_width = input_shape

    for row in range(input_height):
        for column in range(input_width):
            output_array[:, :, row:row + kernel_height, column:column + kernel_width] += engine.sum(inp.data[:, :, row, column] * weight_array[None, :, :, :], axis=(2,))

    return _create_tensor(
        inp,
        weight,
        bias,
        data=output_array + bias_array[:, None, None],
        func=wrapped_partial(conv_transpose_backward, inp=inp, weight=weight, bias=bias, stride=stride, padding=padding),
    )


def bilinear_interpolation(inp, scale_factor):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)
    b, c, height, width = inp.data.shape

    if isinstance(scale_factor, (list, tuple)):  # make sure they have same dimensions
        scale_factors = scale_factor
    else:
        scale_factors = [scale_factor for _ in range(2)]

    output_size = [int(inp.data.shape[idx + 2] * scale_factors[idx]) for idx in range(2)]

    # Initialize the output image
    interpolated_image = engine.zeros([b, c] + output_size, dtype=engine.float32)

    for batch in range(b):
        for y in range(output_size[0]):
            for x in range(output_size[1]):
                # Calculate the corresponding coordinates in the original image
                original_x = x / scale_factors[1]
                original_y = y / scale_factors[0]

                # Find the four nearest neighbor pixels in the original image
                x0 = int(original_x)
                x1 = min(x0 + 1, width - 1)
                y0 = int(original_y)
                y1 = min(y0 + 1, height - 1)

                # Calculate the weights for bilinear interpolation
                weight_x1 = original_x - x0
                weight_x0 = 1 - weight_x1
                weight_y1 = original_y - y0
                weight_y0 = 1 - weight_y1

                # Perform bilinear interpolation for each color channel
                for channel in range(c):
                    interpolated_value = (
                            inp.data[batch, channel, y0, x0] * weight_x0 * weight_y0 +
                            inp.data[batch, channel, y0, x1] * weight_x1 * weight_y0 +
                            inp.data[batch, channel, y1, x0] * weight_x0 * weight_y1 +
                            inp.data[batch, channel, y1, x1] * weight_x1 * weight_y1
                    )
                    interpolated_image[batch, channel, y, x] = interpolated_value

    return _create_tensor(
        inp,
        data=interpolated_image,
        func=wrapped_partial(bilinear_interpolation_backward, inp=inp, scale_factor=scale_factor),
    )


def dropout(inp, keep_prob) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    def apply_mask(array) -> engine.array:
        array *= mask
        array /= keep_prob
        return array

    mask = engine.random.rand(*inp.shape) < keep_prob
    out = apply_mask(inp.data)
    return _create_tensor(
        inp, data=out, func=wrapped_partial(dropout_backward, inp=inp, mask=mask, keep_prob=keep_prob)
    )


def batch_norm(inp, weight, bias, running_mean, running_var, momentum, eps, training) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    input_array = inp.data
    running_mean_array = running_mean.data
    running_var_array = running_var.data
    gamma_array = weight.data
    beta_array = bias.data

    if len(input_array.shape) == 2:
        input_mean = engine.mean(input_array, axis=0)

        input_mean_difference = input_array - input_mean
        input_variance = engine.mean(input_mean_difference ** 2, axis=0)
        input_standard_deviation = engine.sqrt(input_variance)
        input_standard_deviation[input_standard_deviation == 0] = (
                input_standard_deviation[input_standard_deviation == 0] + eps
        )
        input_mean_over_input_standard_deviation = input_mean_difference / input_standard_deviation

        if training:
            input_variance[input_variance == 0] = input_variance[input_variance == 0] + eps
            x_hat = (input_array - input_mean) / engine.sqrt(input_variance)
        else:
            running_var_array[running_var_array == 0] = running_var_array[running_var_array == 0] + eps
            x_hat = (input_array - running_mean_array) / engine.sqrt(running_var_array)
        out = gamma_array * x_hat + beta_array
    elif len(input_array.shape) == 4:
        _, channel, _, _ = input_array.shape
        input_mean = engine.mean(input_array, axis=(0, 2, 3))

        input_mean_difference = input_array - input_mean.reshape((1, channel, 1, 1))
        input_variance = engine.mean(input_mean_difference ** 2, axis=(0, 2, 3))
        input_standard_deviation = engine.sqrt(input_variance.reshape((1, channel, 1, 1)))
        input_standard_deviation[input_standard_deviation == 0] = (
                input_standard_deviation[input_standard_deviation == 0] + eps
        )
        input_mean_over_input_standard_deviation = input_mean_difference / input_standard_deviation

        if training:
            input_variance[input_variance == 0] = input_variance[input_variance == 0] + eps
            x_hat = (input_array - input_mean.reshape((1, channel, 1, 1))) / engine.sqrt(
                input_variance.reshape((1, channel, 1, 1))
            )
        else:
            running_var_array[running_var_array == 0] = running_var_array[running_var_array == 0] + eps
            x_hat = (input_array - running_mean_array.reshape((1, channel, 1, 1))) / engine.sqrt(
                running_var_array.reshape((1, channel, 1, 1))
            )
        out = gamma_array.reshape((1, channel, 1, 1)) * x_hat + beta_array.reshape((1, channel, 1, 1))
    else:
        raise ValueError

    if training:
        running_mean.data = running_mean_array * (1.0 - momentum) + input_mean * momentum
        running_var.data = running_var_array * (1.0 - momentum) + input_variance * momentum

    return _create_tensor(
        inp,
        weight,
        bias,
        data=out,
        func=wrapped_partial(
            batch_norm_backward,
            inp=inp,
            weight=weight,
            bias=bias,
            training=training,
            **{
                'input_standard_deviation': input_standard_deviation,
                'input_mean_difference': input_mean_difference,
                'input_mean_over_input_standard_deviation': input_mean_over_input_standard_deviation,
            }
        ),
    )


def max_pool(inp, kernel_size, stride, padding) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    def save_mask(x, cords):
        mask = engine.zeros_like(x)
        n, c, h, w = x.shape
        x = x.reshape(n, h * w, c)
        idx = engine.argmax(x, axis=1)

        n_idx, c_idx = engine.indices((n, c))
        mask.reshape((n, h * w, c))[n_idx, idx, c_idx] = 1
        cache[cords] = mask

    cache = {}

    padded_input_array = engine.pad(
        inp.data, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode='constant'
    )

    _, _, output_height, output_width = _calculate_output_dims(
        inp.shape, (0, 0, kernel_size[0], kernel_size[1]), padding, stride
    )
    kernel_height, kernel_width = kernel_size
    batch_size, channels, _, _ = padded_input_array.shape

    output_array = engine.zeros((batch_size, channels, output_height, output_width))

    for row in range(output_height):
        for column in range(output_width):
            padded_input_slice = padded_input_array[
                                 :, :, row * stride: row * stride + kernel_height, column * stride: column * stride + kernel_width
                                 ]
            save_mask(x=padded_input_slice, cords=(row, column))
            output_array[:, :, row, column] = engine.amax(padded_input_slice, axis=(2, 3))

    return _create_tensor(
        inp,
        data=output_array,
        func=wrapped_partial(
            max_pool_backward, inp=inp, kernel_size=kernel_size, stride=stride, padding=padding, cache=cache
        ),
    )


def avg_pool(inp, kernel_size, stride, padding) -> 'Tensor':
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(inp)
    engine = _get_engine(inp)

    padded_input_array = engine.pad(
        inp.data, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode='constant'
    )

    _, _, output_height, output_width = _calculate_output_dims(
        inp.shape, (0, 0, kernel_size[0], kernel_size[1]), padding, stride
    )
    kernel_height, kernel_width = kernel_size
    batch_size, channels, _, _ = padded_input_array.shape

    output_array = engine.zeros((batch_size, channels, output_height, output_width))

    for row in range(output_height):
        for column in range(output_width):
            padded_input_slice = padded_input_array[
                                 :, :, row * stride: row * stride + kernel_height, column * stride: column * stride + kernel_width
                                 ]
            output_array[:, :, row, column] = engine.mean(padded_input_slice, axis=(2, 3))

    return _create_tensor(
        inp,
        data=output_array,
        func=wrapped_partial(avg_pool_backward, inp=inp, kernel_size=kernel_size, stride=stride, padding=padding),
    )


def lstm_cell(inp, all_weights):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    batch_size, input_features, hidden_size = inp.size(0), inp.size(1), all_weights[0].size(0) // 4

    engine = _get_engine()

    inp_array = inp.data.copy()

    hx_array = engine.zeros((batch_size, hidden_size))
    cx_array = engine.zeros((batch_size, hidden_size))
    out_array = engine.zeros((batch_size, hidden_size))

    igates = engine.zeros((batch_size, hidden_size))
    fgates = engine.zeros((batch_size, hidden_size))
    cgates = engine.zeros((batch_size, hidden_size))
    ogates = engine.zeros((batch_size, hidden_size))

    all_weights_array = [x.data for x in all_weights]

    cell_input = inp_array[:, :]

    w_ih_array, w_hh_array, b_ih_array, b_hh_array = all_weights_array
    h = hx_array[:, :]  # engine.zeros()
    c = cx_array[:, :]  # engine.zeros()

    gates = cell_input @ w_ih_array.T + b_ih_array + h @ w_hh_array.T + b_hh_array

    ingate, forgetgate, cellgate, outgate = engine.split(gates, 4, 1)

    ingate = 1 / (1 + engine.exp(-ingate))
    forgetgate = 1 / (1 + engine.exp(-forgetgate))
    cellgate = engine.tanh(cellgate)
    outgate = 1 / (1 + engine.exp(-outgate))

    igates[:, :] = ingate
    fgates[:, :] = forgetgate
    cgates[:, :] = cellgate
    ogates[:, :] = outgate

    c = (forgetgate * c) + (ingate * cellgate)
    h = outgate * engine.tanh(c)

    hx_array[:, :] = h
    cx_array[:, :] = c

    out_array[:, :] = h

    cache = {
        'inp': inp_array,
        'out': out_array,
        'hx': hx_array,
        'cx': cx_array,
        'i': igates,
        'f': fgates,
        'c': cgates,
        'o': ogates,
    }

    out_tensor = _create_tensor(
        *([inp] + all_weights),
        data=out_array[:, :],
        func=wrapped_partial(lstm_cell_backward, inp=inp, all_weights=all_weights, cache=cache)
    )
    return out_tensor, (
        from_array(engine.transpose(hx_array[:, :], (1, 0))),
        from_array(engine.transpose(cx_array[:, :], (1, 0))),
    )


def lstm(inp, all_weights):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    # inp.shape = b, t, f
    # hx.shape = n, b, h
    # cx.shape = n, b, h
    # out.shape = b, t, h

    # ilt = σ(Wxihl−1t + Whihlt−1 + bi) (10)
    # flt = σ(Wxfhl−1t + Whfhlt−1 + bf ) (11)
    # olt = σ(Wxohl−1t + Whohlt−1 + bo) (12)
    # zlt = tanh(Wxzhl−1t + Whzhlt−1 + bz ) (13)
    # clt = flt ◦ clt−1 + ilt ◦ zlt(14)
    # hlt = olt ◦ tanh(clt), (15)

    batch_size, time_sequence, input_features, num_layers, hidden_size = (
        inp.size(0),
        inp.size(1),
        inp.size(2),
        len(all_weights),
        all_weights[0][0].size(0) // 4,
    )

    engine = _get_engine()

    inp_array = inp.data.copy()

    # [[[0] * (4 if layer == 0 else 8) for time in range(time)] for layer in range(layer)]

    hx_array = engine.zeros((batch_size, num_layers, time_sequence, hidden_size))
    cx_array = engine.zeros((batch_size, num_layers, time_sequence, hidden_size))
    out_array = engine.zeros((batch_size, num_layers, time_sequence, hidden_size))

    igates = engine.zeros((batch_size, num_layers, time_sequence, hidden_size))
    fgates = engine.zeros((batch_size, num_layers, time_sequence, hidden_size))
    cgates = engine.zeros((batch_size, num_layers, time_sequence, hidden_size))
    ogates = engine.zeros((batch_size, num_layers, time_sequence, hidden_size))

    w_ih_arrays = [x[0].data for x in all_weights]
    w_hh_arrays = [x[1].data for x in all_weights]
    b_ih_arrays = [x[2].data for x in all_weights]
    b_hh_arrays = [x[3].data for x in all_weights]

    for time in range(time_sequence):
        for layer in range(num_layers):
            if time == 0:
                if layer == 0:
                    cell_input = inp_array[:, time, :]
                else:
                    cell_input = out_array[:, layer - 1, time, :]
            else:
                if layer == 0:
                    cell_input = inp_array[:, time, :]
                else:
                    cell_input = out_array[:, layer - 1, time, :]

            w_ih_array, w_hh_array, b_ih_array, b_hh_array = (
                w_ih_arrays[layer],
                w_hh_arrays[layer],
                b_ih_arrays[layer],
                b_hh_arrays[layer],
            )

            if time == 0:
                h = hx_array[:, layer, 0, :]  # engine.zeros()
                c = cx_array[:, layer, 0, :]  # engine.zeros()
            else:
                h = hx_array[:, layer, time - 1, :]
                c = cx_array[:, layer, time - 1, :]

            gates = cell_input @ w_ih_array.T + b_ih_array + h @ w_hh_array.T + b_hh_array

            ingate, forgetgate, cellgate, outgate = engine.split(gates, 4, 1)

            ingate = 1 / (1 + engine.exp(-ingate))
            forgetgate = 1 / (1 + engine.exp(-forgetgate))
            cellgate = engine.tanh(cellgate)
            outgate = 1 / (1 + engine.exp(-outgate))

            igates[:, layer, time, :] = ingate
            fgates[:, layer, time, :] = forgetgate
            cgates[:, layer, time, :] = cellgate
            ogates[:, layer, time, :] = outgate

            c = (forgetgate * c) + (ingate * cellgate)
            h = outgate * engine.tanh(c)

            hx_array[:, layer, time, :] = h
            cx_array[:, layer, time, :] = c

            out_array[:, layer, time, :] = h

    cache = {
        'inp': inp_array,
        'out': out_array,
        'hx': hx_array,
        'cx': cx_array,
        'i': igates,
        'f': fgates,
        'c': cgates,
        'o': ogates,
    }

    out_tensor = _create_tensor(
        *([inp] + sum([layer_weights for layer_weights in all_weights], [])),
        data=out_array[:, num_layers - 1, :, :],
        func=wrapped_partial(lstm_backward, inp=inp, all_weights=all_weights, cache=cache)
    )
    return out_tensor, (
        from_array(engine.transpose(hx_array[:, :, time_sequence - 1, :], (1, 0, 2))),
        from_array(engine.transpose(cx_array[:, :, time_sequence - 1, :], (1, 0, 2))),
    )


def adam(
        params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, amsgrad, beta1, beta2, lr, weight_decay, eps
):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(*params)
    engine = _get_engine(*params)

    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad._data = grad.data + param.data * weight_decay

        # Decay the first and second moment running average coefficient
        exp_avg._data = exp_avg.data * beta1 + (1 - beta1) * grad.data
        # check if this is true
        exp_avg_sq._data = exp_avg_sq.data * beta2
        exp_avg_sq._data = exp_avg_sq.data + (1 - beta2) * (grad.data * grad.data)
        # exp_avg_sq._data = exp_avg_sq.data * beta2 + (1 - beta2) * (grad.data * grad.data)
        if amsgrad:
            max_exp_avg_sq = max_exp_avg_sqs[i]
            # Maintains the maximum of all 2nd moment running avg. till now
            max_exp_avg_sq._data = engine.maximum(max_exp_avg_sq.data, exp_avg_sq.data)
            # Use the max. for normalizing running avg. of gradient
            denom = (engine.sqrt(max_exp_avg_sq.data) / math.sqrt(bias_correction2)) + eps
        else:
            denom = (engine.sqrt(exp_avg_sq.data) / math.sqrt(bias_correction2)) + eps

        step_size = lr / bias_correction1

        param._data = param.data - step_size * (exp_avg.data / denom)


def rmsprop(params, grads, square_avgs, alphas, momentum_buffers, grad_avgs, momentum, centered, lr, weight_decay, eps):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    _check_tensors(*params)
    engine = _get_engine(*params)

    for i, param in enumerate(params):
        grad = grads[i]
        square_avg = square_avgs[i]
        alpha = alphas[i]

        if weight_decay != 0:
            grad._data = grad.data + param.data * weight_decay

        square_avg._data = square_avg.data * alpha + (1 - alpha) * (grad.data * grad.data)

        if centered:
            grad_avg = grad_avgs[i]
            grad_avg._data = grad_avg.data * alpha + grad.data * (1 - alpha)
            avg = engine.sqrt(square_avg.data - (grad_avg.data * grad_avg.data)) + eps
        else:
            avg = engine.sqrt(square_avg.data) + eps

        if momentum > 0:
            buf = momentum_buffers[i]
            # check if this is true
            buf._data = buf.data * momentum
            buf._data = buf.data + grad.data / avg
            # buf._data = buf.data * momentum + grad.data / avg
            param._data = param.data - lr * buf.data
        else:
            param._data = param.data - lr * grad.data / avg
