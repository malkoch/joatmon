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

from joatmon.ai.nn.core import Tensor
from joatmon.ai.nn.utility import _calculate_output_dims

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
    'is_tensor', 'concat_backward', 'stack_backward', 'chunk_backward', 'view_backward', 'index_select_backward',
    'squeeze_backward', 'expand_dim_backward', 'transpose_backward', 'absolute_backward', 'around_backward',
    'floor_backward', 'ceil_backward', 'clip_backward', 'negative_backward', 'summation_backward', 'mean_backward',
    'std_backward', 'var_backward', 'add_backward', 'sub_backward', 'mul_backward', 'div_backward', 'power_backward',
    'clone_backward', 'relu_backward', 'sigmoid_backward', 'softmax_backward', 'tanh_backward', 'dense_backward',
    'conv_backward', 'dropout_backward', 'batch_norm_backward', 'max_pool_backward', 'avg_pool_backward',
    'rnn_relu_backward', 'rnn_tanh_backward', 'lstm_backward', 'gru_backward', 'rnn_relu_cell_backward',
    'rnn_tanh_cell_backward', 'lstm_cell_backward', 'gru_cell_backward', 'concat', 'stack', 'chunk', 'view',
    'index_select', 'zero', 'one', 'fill', 'squeeze', 'expand_dim', 'transpose', 'absolute', 'around', 'floor',
    'ceil', 'clip', 'negative', 'summation', 'mean', 'std', 'var', 'add', 'sub', 'mul', 'div', 'power', 'clone',
    'detach', 'arange', 'linspace', 'normal', 'uniform', 'rand', 'randint', 'randn', 'eye', 'empty', 'full', 'zeros',
    'ones', 'normal_like', 'uniform_like', 'rand_like', 'randint_like', 'randn_like', 'eye_like', 'empty_like',
    'full_like', 'zeros_like', 'ones_like', 'from_array', 'to_array', 'half', 'single', 'double', 'cpu', 'gpu',
    'relu', 'sigmoid', 'softmax', 'tanh', 'dense', 'conv', 'dropout', 'batch_norm', 'max_pool', 'avg_pool',
    'rnn_relu', 'rnn_tanh', 'lstm', 'gru', 'rnn_relu_cell', 'rnn_tanh_cell', 'lstm_cell', 'gru_cell', 'adam', 'rmsprop'
]


# need to implement inplace

# should have c / c++ codes to use them in functional apis


def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def _check_tensor_devices(*tensors: Tensor):
    iterator = iter(tensors)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first.device == x.device for x in iterator)


def _check_tensors(*tensors: Tensor):
    if not _check_tensor_devices(*tensors):
        raise ValueError('devices are not matching')

    if len(tensors) == 0:
        raise ValueError('there should be at least one tensor')

    if tensors[0].device not in ('cpu', 'gpu'):
        raise ValueError('device has to be either \'cpu\' or \'gpu\'')


def _get_engine(*tensors: Union[Tensor, str]):
    import numpy as engine
    return engine


def _set_grad(tensor: Tensor, data):
    if not (tensor.requires_grad and not hasattr(tensor, "retains_grad") and not tensor.is_leaf):
        if not tensor.is_leaf:
            return

        if tensor.grad is None:
            tensor.grad = from_array(data, device=tensor.device)
        else:
            tensor.grad._data = tensor.grad.data + data
    tensor.backward(data)


def _create_tensor(*tensors: Tensor, data, func):
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
    return isinstance(obj, Tensor)


def concat_backward(gradient: Tensor, tensors: List[Tensor], axis: int = 0):
    _check_tensors(*tensors)
    engine = _get_engine(*tensors)

    grad_arrays = engine.split(gradient.data, len(tensors), axis=axis)
    for idx, tensor in enumerate(tensors):
        _set_grad(tensor, data=grad_arrays[idx] * engine.ones_like(tensor.data))


def stack_backward(gradient: Tensor, tensors: List[Tensor], axis: int = 0):
    _check_tensors(*tensors)
    engine = _get_engine(*tensors)

    grad_arrays = engine.split(gradient.data, len(tensors), axis=axis)
    for idx, tensor in enumerate(tensors):
        _set_grad(tensor, data=grad_arrays[idx] * engine.ones_like(tensor.data))


def chunk_backward(gradient: Tensor, tensor: Tensor, chunks: int):
    _check_tensors(tensor)
    engine = _get_engine(tensor)

    _set_grad(tensor, gradient.data * engine.ones_like(tensor.data) / chunks)


def view_backward(gradient: Tensor, inp: Tensor):
    _check_tensors(inp)

    _set_grad(inp, gradient.data.reshape(inp.shape))


def index_select_backward(gradient: Tensor, inp: Tensor, index: Tensor, dim: int):
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
    _check_tensors(inp)
    engine = _get_engine(inp)

    _set_grad(inp, gradient.data * engine.ones_like(inp.data))


def expand_dim_backward(gradient: Tensor, inp: Tensor):
    _check_tensors(inp)
    engine = _get_engine(inp)

    _set_grad(inp, gradient.data * engine.ones_like(inp.data))


def transpose_backward(gradient: Tensor, inp: Tensor):
    _check_tensors(inp)
    engine = _get_engine(inp)

    _set_grad(inp, gradient.data * engine.ones_like(inp.data))


def absolute_backward(gradient: Tensor, inp: Tensor):
    _check_tensors(inp)
    engine = _get_engine(inp)

    _set_grad(inp, gradient.data * engine.ones_like(inp.data))


def around_backward(gradient: Tensor, inp: Tensor):
    _check_tensors(inp)
    engine = _get_engine(inp)

    _set_grad(inp, gradient.data * engine.ones_like(inp.data))


def floor_backward(gradient: Tensor, inp: Tensor):
    _check_tensors(inp)
    engine = _get_engine(inp)

    _set_grad(inp, gradient.data * engine.ones_like(inp.data))


def ceil_backward(gradient: Tensor, inp: Tensor):
    _check_tensors(inp)
    engine = _get_engine(inp)

    _set_grad(inp, gradient.data * engine.ones_like(inp.data))


def clip_backward(gradient: Tensor, inp: Tensor):
    _check_tensors(inp)
    engine = _get_engine(inp)

    _set_grad(inp, gradient.data * engine.ones_like(inp.data))


def negative_backward(gradient: Tensor, inp: Tensor):
    _check_tensors(inp)
    engine = _get_engine(inp)

    _set_grad(inp, gradient.data * engine.ones_like(inp.data))


def summation_backward(gradient: Tensor, inp: Tensor):
    _check_tensors(inp)
    engine = _get_engine(inp)

    _set_grad(inp, gradient.data * engine.ones_like(inp.data))


def mean_backward(gradient: Tensor, inp: Tensor):
    _check_tensors(inp)
    engine = _get_engine(inp)

    _set_grad(inp, gradient.data * engine.ones_like(inp.data))


def std_backward(gradient: Tensor, inp: Tensor):
    _check_tensors(inp)
    engine = _get_engine(inp)

    _set_grad(inp, gradient.data * engine.ones_like(inp.data))


def var_backward(gradient: Tensor, inp: Tensor):
    _check_tensors(inp)
    engine = _get_engine(inp)

    _set_grad(inp, gradient.data * engine.ones_like(inp.data))


def add_backward(gradient: Tensor, inp1: Tensor, inp2: Tensor):
    _check_tensors(inp1, inp2)
    engine = _get_engine(inp1, inp2)

    _set_grad(inp1, gradient.data * engine.ones_like(inp1.data))
    _set_grad(inp2, gradient.data * engine.ones_like(inp2.data))


def sub_backward(gradient: Tensor, inp1: Tensor, inp2: Tensor):
    _check_tensors(inp1, inp2)
    engine = _get_engine(inp1, inp2)

    _set_grad(inp1, gradient.data * engine.ones_like(inp1.data))
    _set_grad(inp2, gradient.data * -engine.ones_like(inp2.data))


def mul_backward(gradient: Tensor, inp1: Tensor, inp2: Tensor):
    _check_tensors(inp1, inp2)

    _set_grad(inp1, gradient.data * inp2.data)
    _set_grad(inp2, gradient.data * inp1.data)


def div_backward(gradient: Tensor, inp1: Tensor, inp2: Tensor):
    _check_tensors(inp1, inp2)

    _set_grad(inp1, gradient.data * (1 / inp2.data))
    _set_grad(inp2, gradient.data * inp1.data)


def power_backward(gradient: Tensor, inp: Tensor, p: int):
    _check_tensors(inp)

    _set_grad(inp, gradient.data * p * (inp.data ** (p - 1)))


def clone_backward(gradient: Tensor, inp: Tensor):
    _check_tensors(inp)
    engine = _get_engine(inp)

    _set_grad(inp, gradient.data * engine.ones_like(inp.data))


def relu_backward(gradient: Tensor, inp: Tensor):
    _check_tensors(inp)
    engine = _get_engine(inp)

    out = engine.zeros_like(inp.data)
    out[inp.data <= 0] = 0
    out[inp.data > 0] = 1

    _set_grad(inp, gradient.data * out)


def sigmoid_backward(gradient: Tensor, inp: Tensor, out):
    _check_tensors(inp)

    _set_grad(inp, gradient.data * out * (1 - out))


def softmax_backward(gradient: Tensor, inp: Tensor):
    _check_tensors(inp)
    engine = _get_engine(inp)

    grad_array = gradient.data

    indices = engine.where(grad_array == grad_array.max())

    arr = -grad_array * grad_array
    arr[indices] = grad_array[indices] * (1 - grad_array[indices])

    _set_grad(inp, arr)


def tanh_backward(gradient: Tensor, inp: Tensor, out: Tensor):
    _check_tensors(inp)
    engine = _get_engine(inp)

    _set_grad(inp, gradient.data * (1 - engine.square(out.data)))


def dense_backward(gradient: Tensor, inp: Tensor, weight: Tensor, bias: Tensor):
    _check_tensors(inp, weight, bias)
    engine = _get_engine(inp, weight, bias)

    _set_grad(inp, engine.dot(gradient.data, weight.data))
    _set_grad(weight, engine.dot(gradient.data.T, inp.data))
    _set_grad(bias, engine.sum(gradient.data, axis=0, keepdims=True))


def conv_backward(gradient: Tensor, inp: Tensor, weight: Tensor, bias: Tensor, stride: int, padding: Union[List[int], Tuple[int]]):
    _check_tensors(inp, weight, bias)
    engine = _get_engine(inp, weight, bias)

    _padded_input_array = engine.pad(inp.data, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode='constant')
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
            _output_array[:, :, _row * stride:_row * stride + _kernel_height, _column * stride:_column * stride + _kernel_width] += engine.sum(
                _weight_array[None, :, :, :, :] *
                _grad_array[:, :, None, _row:_row + 1, _column:_column + 1],
                axis=1
            )
            _weight_grad += engine.sum(
                _padded_input_array[:, None, :, _row * stride:_row * stride + _kernel_height, _column * stride:_column * stride + _kernel_width] *
                _grad_array[:, :, None, _row:_row + 1, _column:_column + 1],
                axis=0
            )

    _set_grad(inp, _weight_grad)
    _set_grad(weight, _bias_grad)
    _set_grad(bias, _output_array[:, :, padding[0]:padding[0] + _input_height, padding[1]:padding[1] + input_width])


def dropout_backward(gradient: Tensor, inp: Tensor, mask, keep_prob: float):
    _check_tensors(inp)
    engine = _get_engine(inp)

    def apply_mask(array) -> engine.array:
        array *= mask
        array /= keep_prob
        return array

    _set_grad(inp, apply_mask(gradient))


def batch_norm_backward(gradient: Tensor, inp: Tensor, weight: Tensor, bias: Tensor, training: bool, **kwargs):
    _check_tensors(inp)
    engine = _get_engine(inp)

    if training:
        batch_size = inp.data.shape[0]
        weight_by_grad = weight.data * gradient.data
        dxc = weight_by_grad / kwargs['input_standard_deviation']
        dstd = -engine.sum(
            (weight_by_grad * kwargs['input_mean_difference']) / (kwargs['input_standard_deviation'] * kwargs['input_standard_deviation']),
            axis=0
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


def max_pool_backward(gradient: Tensor, inp: Tensor, kernel_size: Union[List[int], Tuple[int]], stride: int, padding: Union[List[int], Tuple[int]], cache: dict):
    _check_tensors(inp)
    engine = _get_engine(inp)

    grad_array = gradient.data

    _, _, _output_height, _output_width = grad_array.shape
    _kernel_height, _kernel_width = kernel_size

    _padded_input_array = engine.pad(inp.data, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode='constant')
    _output_array = engine.zeros_like(_padded_input_array)

    for _row in range(_output_height):
        for _column in range(_output_width):
            increment = grad_array[:, :, _row:_row + 1, _column:_column + 1] * cache[(_row, _column)]
            _output_array[:, :, _row * stride:_row * stride + _kernel_height, _column * stride:_column * stride + _kernel_width] += increment

    _set_grad(inp, _output_array[:, :, padding[0]:padding[0] + _output_height - 1, padding[1]:padding[1] + _output_width - 1])


def avg_pool_backward(gradient: Tensor, inp: Tensor, kernel_size: Union[List[int], Tuple[int]], stride: int, padding: Union[List[int], Tuple[int]]):
    _check_tensors(inp)
    engine = _get_engine(inp)

    grad_array = gradient.data

    _, _, _output_height, _output_width = grad_array.shape
    _kernel_height, _kernel_width = kernel_size

    _padded_input_array = engine.pad(inp.data, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode='constant')
    _output_array = engine.zeros_like(_padded_input_array)

    for _row in range(_output_height):
        for _column in range(_output_width):
            increment = grad_array[:, :, _row:_row + 1, _column:_column + 1] / _kernel_height / _kernel_width
            _output_array[:, :, _row * stride:_row * stride + _kernel_height, _column * stride:_column * stride + _kernel_width] += increment

    _set_grad(inp, _output_array[:, :, padding[0]:padding[0] + _output_height - 1, padding[1]:padding[1] + _output_width - 1])


def rnn_relu_backward(gradient: Tensor, inp: Tensor, all_weights: List[Union[List[Tensor], Tuple[Tensor]]], num_layers: int, intermediate_values: dict):
    _check_tensors(inp)
    engine = _get_engine(inp)

    _inp_grads = engine.zeros_like(inp.data)

    # for each layer need to create new gradient array
    # from last layer to first layer
    for _layer in range(num_layers):
        _w_ih, _w_hh, _b_ih, _b_hh = all_weights[_layer]

        _w_ih_grads = engine.zeros_like(_w_ih.data)
        _w_hh_grads = engine.zeros_like(_w_hh.data)
        _b_ih_grads = engine.zeros_like(_b_ih.data)
        _b_hh_grads = engine.zeros_like(_b_hh.data)

        for _time in range(inp.size(1) - 1, -1, -1):
            _prev_h = intermediate_values[_time][_layer]['prev_h']
            _current_h = intermediate_values[_time][_layer]['current_h']
            _inp = intermediate_values[_time][_layer]['input']
            _relu_input = intermediate_values[_time][_layer]['relu_input']

            _w_ih_power = engine.power(_w_ih.data, inp.size(1) - _time - 1)
            _w_hh_power = engine.power(_w_hh.data, inp.size(1) - _time - 1)

            _out = engine.zeros_like(_relu_input.data)
            _out[_relu_input.data <= 0] = 0
            _out[_relu_input.data > 0] = 1

            _w_ih_grads += engine.dot((_out * gradient.data[:, _time, :]).T, _inp.data) * _w_ih_power
            _w_hh_grads += engine.dot((_out * gradient.data[:, _time, :]).T, _prev_h.data) * _w_hh_power
            _b_ih_grads += engine.sum((_out * gradient.data[:, _time, :]), axis=0)
            _b_hh_grads += engine.sum((_out * gradient.data[:, _time, :]), axis=0)

            _inp_grads[:, _time, :] += engine.dot((_out * gradient.data[:, _time, :]), _w_ih.data)

        _set_grad(_w_ih, data=_w_ih_grads)
        _set_grad(_w_hh, data=_w_hh_grads)
        _set_grad(_b_ih, data=_b_ih_grads)
        _set_grad(_b_hh, data=_b_hh_grads)
    _set_grad(inp, data=_inp_grads)


def rnn_tanh_backward(gradient, inp, hx, all_weights, bias, num_layers):
    ...


def lstm_backward(gradient, inp, hx, all_weights, bias, num_layers):
    ...


def gru_backward(gradient, inp, hx, all_weights, bias, num_layers):
    ...


def rnn_relu_cell_backward(gradient, inp, h, w_ih, w_hh, b_ih=None, b_hh=None):
    ...


def rnn_tanh_cell_backward(gradient, inp, h, w_ih, w_hh, b_ih=None, b_hh=None):
    ...


def lstm_cell_backward(gradient, inp, h, w_ih, w_hh, b_ih=None, b_hh=None):
    ...


def gru_cell_backward(gradient):
    ...


def concat(tensors: List[Tensor], axis: int = 0) -> 'Tensor':
    _check_tensors(*tensors)
    engine = _get_engine(*tensors)

    return _create_tensor(
        *tensors,
        data=engine.concatenate(list(map(lambda x: x.data, tensors)), axis=axis),
        func=wrapped_partial(concat_backward, tensors=tensors, axis=axis)
    )


def stack(tensors: List[Tensor], axis: int = 0) -> 'Tensor':
    _check_tensors(*tensors)
    engine = _get_engine(*tensors)

    return _create_tensor(
        *tensors,
        data=engine.stack(list(map(lambda x: x.data, tensors)), axis=axis),
        func=wrapped_partial(stack_backward, tensors=tensors, axis=axis)
    )


def chunk(tensor: Tensor, chunks: int, dim: int = 0):
    _check_tensors(tensor)
    engine = _get_engine(tensor)

    arrays = engine.split(tensor.data, chunks, dim)

    tensors = []
    for array in arrays:
        tensors.append(_create_tensor(
            tensor,
            data=array,
            func=wrapped_partial(chunk_backward, tensor=tensor, chunks=chunks)
        ))
    return tensors


def view(inp, size=None) -> 'Tensor':
    _check_tensors(inp)

    return _create_tensor(
        inp,
        data=inp.data.reshape(size),
        func=wrapped_partial(view_backward, inp)
    )


def index_select(inp, dim, index) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)

    return _create_tensor(
        inp,
        data=engine.take_along_axis(inp.data, index.data.astype('int'), dim),
        func=wrapped_partial(index_select_backward, inp=inp, index=index, dim=dim)
    )


def zero(inp) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)

    inp._data = engine.zeros_like(inp.data)
    return inp


def one(inp) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)

    inp._data = engine.ones_like(inp.data)
    return inp


def fill(inp, value) -> 'Tensor':
    _check_tensors(inp)

    inp.data.fill(value)
    return inp


def squeeze(inp, axis=None) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)

    return _create_tensor(
        inp,
        data=engine.squeeze(inp.data, axis=axis),
        func=wrapped_partial(squeeze_backward, inp=inp)
    )


def expand_dim(inp, axis=None) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)

    return _create_tensor(
        inp,
        data=engine.expand_dims(inp.data, axis=axis),
        func=wrapped_partial(expand_dim_backward, inp=inp)
    )


def transpose(inp, axes=None) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)

    return _create_tensor(
        inp,
        data=engine.transpose(inp.data, axes=axes),
        func=wrapped_partial(transpose_backward, inp=inp)
    )


def absolute(inp) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)

    return _create_tensor(
        inp,
        data=engine.absolute(inp.data),
        func=wrapped_partial(absolute_backward, inp=inp)
    )


def around(inp) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)

    return _create_tensor(
        inp,
        data=engine.around(inp.data),
        func=wrapped_partial(around_backward, inp=inp)
    )


def floor(inp) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)

    return _create_tensor(
        inp,
        data=engine.floor(inp.data),
        func=wrapped_partial(floor_backward, inp=inp)
    )


def ceil(inp) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)

    return _create_tensor(
        inp,
        data=engine.ceil(inp.data),
        func=wrapped_partial(ceil_backward, inp=inp)
    )


def clip(inp, min_val, max_val) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)

    return _create_tensor(
        inp,
        data=engine.clip(inp.data, min_val, max_val),
        func=wrapped_partial(clip_backward, inp=inp)
    )


def negative(inp) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)

    return _create_tensor(
        inp,
        data=engine.negative(inp.data),
        func=wrapped_partial(negative_backward, inp=inp)
    )


def summation(inp) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)

    return _create_tensor(
        inp,
        data=engine.sum(inp.data),
        func=wrapped_partial(summation_backward, inp=inp)
    )


def mean(inp) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)

    return _create_tensor(
        inp,
        data=engine.mean(inp.data),
        func=wrapped_partial(mean_backward, inp=inp)
    )


def std(inp) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)

    return _create_tensor(
        inp,
        data=engine.std(inp.data),
        func=wrapped_partial(std_backward, inp=inp)
    )


def var(inp) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)

    return _create_tensor(
        inp,
        data=engine.var(inp.data),
        func=wrapped_partial(var_backward, inp=inp)
    )


def add(inp1, inp2) -> 'Tensor':
    if not isinstance(inp2, Tensor):
        inp2 = from_array(inp2, device=inp1.device)

    _check_tensors(inp1, inp2)

    return _create_tensor(
        inp1,
        inp2,
        data=inp1.data + inp2.data,
        func=wrapped_partial(add_backward, inp1=inp1, inp2=inp2)
    )


def sub(inp1, inp2) -> 'Tensor':
    if not isinstance(inp2, Tensor):
        inp2 = from_array(inp2, device=inp1.device)

    _check_tensors(inp1, inp2)

    return _create_tensor(
        inp1,
        inp2,
        data=inp1.data - inp2.data,
        func=wrapped_partial(sub_backward, inp1=inp1, inp2=inp2)
    )


def mul(inp1, inp2) -> 'Tensor':
    if not isinstance(inp2, Tensor):
        inp2 = from_array(inp2, device=inp1.device)

    _check_tensors(inp1, inp2)

    return _create_tensor(
        inp1,
        inp2,
        data=inp1.data * inp2.data,
        func=wrapped_partial(mul_backward, inp1=inp1, inp2=inp2)
    )


def div(inp1, inp2) -> 'Tensor':
    if not isinstance(inp2, Tensor):
        inp2 = from_array(inp2, device=inp1.device)

    _check_tensors(inp1, inp2)

    return _create_tensor(
        inp1,
        inp2,
        data=inp1.data / inp2.data,
        func=wrapped_partial(div_backward, inp1=inp1, inp2=inp2)
    )


def power(inp, p) -> 'Tensor':
    # if not isinstance(p, Tensor):
    #     p = from_array(p, device=inp.device)

    _check_tensors(inp)

    return _create_tensor(
        inp,
        data=inp.data ** p,
        func=wrapped_partial(power_backward, inp=inp, p=p)
    )


def clone(inp) -> 'Tensor':
    _check_tensors(inp)

    return _create_tensor(
        inp,
        data=inp.data,
        func=wrapped_partial(clone_backward, inp=inp)
    )


def detach(inp, inplace=True) -> 'Tensor':
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
    engine = _get_engine(device)
    return from_array(engine.arange(start, stop, step).astype(dtype), requires_grad, device)


def linspace(start, end, steps, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    engine = _get_engine(device)
    return from_array(engine.linspace(start, end, steps).astype(dtype), requires_grad, device)


def normal(loc=0.0, scale=1.0, size=None, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    engine = _get_engine(device)
    return from_array(engine.random.normal(loc, scale, size).astype(dtype), requires_grad, device)


def uniform(low=-1.0, high=1.0, size=None, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    engine = _get_engine(device)
    return from_array(engine.random.uniform(low, high, size).astype(dtype), requires_grad, device)


def rand(size, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    engine = _get_engine(device)
    return from_array(engine.random.rand(size).astype(dtype), requires_grad, device)


def randint(low=0, high=0, size=None, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    engine = _get_engine(device)
    return from_array(engine.random.randint(low, high, *size).astype(dtype), requires_grad, device)


def randn(size, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    engine = _get_engine(device)
    return from_array(engine.random.randn(size).astype(dtype), requires_grad, device)


def eye(rows, columns=None, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    engine = _get_engine(device)
    return from_array(engine.eye(rows, columns).astype(dtype), requires_grad, device)


def empty(size, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    engine = _get_engine(device)
    return from_array(engine.empty(size).astype(dtype), requires_grad, device)


def full(size, fill_value, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    engine = _get_engine(device)
    return from_array(engine.full(size, fill_value).astype(dtype), requires_grad, device)


def zeros(size, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    engine = _get_engine(device)
    return from_array(engine.zeros(size).astype(dtype), requires_grad, device)


def ones(size, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    engine = _get_engine(device)
    return from_array(engine.ones(size).astype(dtype), requires_grad, device)


def normal_like(tensor, loc=0.0, scale=1.0, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    return normal(loc, scale, tensor.shape, requires_grad, device, dtype)


def uniform_like(tensor, low=-1.0, high=1.0, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    return uniform(low, high, tensor.shape, requires_grad, device, dtype)


def rand_like(tensor, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    return rand(tensor.shape, requires_grad, device, dtype)


def randint_like(tensor, low=0, high=0, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    return randint(low, high, tensor.shape, requires_grad, device, dtype)


def randn_like(tensor, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    return randn(tensor.shape, requires_grad, device, dtype)


def eye_like(tensor, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    return eye(tensor.shape[0], tensor.shape[1], requires_grad, device, dtype)


def empty_like(tensor, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    return empty(tensor.shape, requires_grad, device, dtype)


def full_like(tensor, fill_value, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    return full(tensor.shape, fill_value, requires_grad, device, dtype)


def zeros_like(tensor, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    return zeros(tensor.shape, requires_grad, device, dtype)


def ones_like(tensor, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    return ones(tensor.shape, requires_grad, device, dtype)


def from_array(data, requires_grad=False, device='cpu', dtype='float32') -> 'Tensor':
    engine = _get_engine('cpu')
    tensor = Tensor(engine.copy(data).astype(dtype), requires_grad=requires_grad)
    if device == 'gpu':
        tensor = tensor.gpu()
    return tensor


def to_array(inp):
    _check_tensors(inp)
    engine = _get_engine('cpu')

    if inp.device != 'cpu':
        raise TypeError('can\'t convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.')
    if inp.requires_grad:
        raise RuntimeError('Can\'t call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.')
    return engine.array(inp.data, copy=True)


def half(inp) -> 'Tensor':
    inp._data = inp.data.astype('float16')
    return inp


def single(inp) -> 'Tensor':
    inp._data = inp.data.astype('float32')
    return inp


def double(inp) -> 'Tensor':
    inp._data = inp.data.astype('float64')
    return inp


def cpu(inp) -> 'Tensor':
    if inp.device == 'cpu':
        return inp

    inp._device = 'cpu'
    inp._data = _get_engine().array(inp.data)
    # inp._data = cp.asnumpy(inp.data)
    return inp


def gpu(inp) -> 'Tensor':
    if inp.device == 'gpu':
        return inp

    inp._device = 'gpu'
    inp._data = _get_engine().array(inp.data)
    # inp._data = cp.array(inp.data)
    return inp


def relu(inp) -> 'Tensor':
    _check_tensors(inp)

    arr = inp.data
    arr[arr <= 0] = 0
    return _create_tensor(
        inp,
        data=arr,
        func=wrapped_partial(relu_backward, inp=inp)
    )


def sigmoid(inp) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)

    output_array = 1 / (1 + engine.exp(-inp.data))
    return _create_tensor(
        inp,
        data=output_array,
        func=wrapped_partial(sigmoid_backward, inp=inp, out=output_array)
    )


def softmax(inp) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)

    e = engine.exp(inp.data - inp.data.amax(axis=1, keepdims=True))
    z = e / engine.sum(e, axis=1, keepdims=True)
    return _create_tensor(
        inp,
        data=z,
        func=wrapped_partial(softmax_backward, inp=inp)
    )


def tanh(inp) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)

    output_array = engine.tanh(inp.data)
    return _create_tensor(
        inp,
        data=output_array,
        func=wrapped_partial(tanh_backward, inp=inp, out=output_array)
    )


def dense(inp, weight, bias) -> 'Tensor':
    _check_tensors(inp, weight, bias)
    engine = _get_engine(inp, weight, bias)

    return _create_tensor(
        inp,
        weight,
        bias,
        data=engine.dot(inp.data, weight.data.T) + bias.data,
        func=wrapped_partial(dense_backward, inp=inp, weight=weight, bias=bias)
    )


def conv(inp, weight, bias, stride, padding) -> 'Tensor':
    _check_tensors(inp, weight, bias)
    engine = _get_engine(inp, weight, bias)

    padded_input_array = engine.pad(inp.data, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode='constant')
    weight_array = weight.data
    bias_array = bias.data

    output_shape = _calculate_output_dims(
        input_shape=inp.shape,
        kernel_shape=weight.shape,
        padding=padding,
        stride=stride
    )

    output_array = engine.zeros(output_shape)

    _, _, kernel_height, kernel_width = weight.shape
    _, _, output_height, output_width = output_shape

    for row in range(output_height):
        for column in range(output_width):
            output_array[:, :, row, column] = engine.sum(
                padded_input_array[:, None, :, row * stride:row * stride + kernel_height, column * stride:column * stride + kernel_width] *
                weight_array[None, :, :, :],
                axis=(2, 3, 4)
            )

    return _create_tensor(
        inp,
        weight,
        bias,
        data=output_array + bias_array[:, None, None],
        func=wrapped_partial(conv_backward, inp=inp, weight=weight, bias=bias, stride=stride, padding=padding)
    )


def dropout(inp, keep_prob) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)

    def apply_mask(array) -> engine.array:
        array *= mask
        array /= keep_prob
        return array

    mask = (engine.random.rand(*inp.shape) < keep_prob)
    out = apply_mask(inp.data)
    return _create_tensor(
        inp,
        data=out,
        func=wrapped_partial(dropout_backward, inp=inp, mask=mask, keep_prob=keep_prob)
    )


def batch_norm(inp, weight, bias, running_mean, running_var, momentum, eps, training) -> 'Tensor':
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
        input_standard_deviation[input_standard_deviation == 0] = input_standard_deviation[input_standard_deviation == 0] + eps
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
        input_standard_deviation[input_standard_deviation == 0] = input_standard_deviation[input_standard_deviation == 0] + eps
        input_mean_over_input_standard_deviation = input_mean_difference / input_standard_deviation

        if training:
            input_variance[input_variance == 0] = input_variance[input_variance == 0] + eps
            x_hat = (input_array - input_mean.reshape((1, channel, 1, 1))) / engine.sqrt(input_variance.reshape((1, channel, 1, 1)))
        else:
            running_var_array[running_var_array == 0] = running_var_array[running_var_array == 0] + eps
            x_hat = (input_array - running_mean_array.reshape((1, channel, 1, 1))) / engine.sqrt(running_var_array.reshape((1, channel, 1, 1)))
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
        func=wrapped_partial(batch_norm_backward, inp=inp, weight=weight, bias=bias, training=training, **{
            'input_standard_deviation': input_standard_deviation,
            'input_mean_difference': input_mean_difference,
            'input_mean_over_input_standard_deviation': input_mean_over_input_standard_deviation
        }
                             )
    )


def max_pool(inp, kernel_size, stride, padding) -> 'Tensor':
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

    padded_input_array = engine.pad(inp.data, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode='constant')

    _, _, output_height, output_width = _calculate_output_dims(inp.shape, (0, 0, kernel_size[0], kernel_size[1]), padding, stride)
    kernel_height, kernel_width = kernel_size
    batch_size, channels, _, _ = padded_input_array.shape

    output_array = engine.zeros((batch_size, channels, output_height, output_width))

    for row in range(output_height):
        for column in range(output_width):
            padded_input_slice = padded_input_array[:, :, row * stride:row * stride + kernel_height, column * stride:column * stride + kernel_width]
            save_mask(x=padded_input_slice, cords=(row, column))
            output_array[:, :, row, column] = engine.amax(padded_input_slice, axis=(2, 3))

    return _create_tensor(
        inp,
        data=output_array,
        func=wrapped_partial(max_pool_backward, inp=inp, kernel_size=kernel_size, stride=stride, padding=padding, cache=cache)
    )


def avg_pool(inp, kernel_size, stride, padding) -> 'Tensor':
    _check_tensors(inp)
    engine = _get_engine(inp)

    padded_input_array = engine.pad(inp.data, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode='constant')

    _, _, output_height, output_width = _calculate_output_dims(inp.shape, (0, 0, kernel_size[0], kernel_size[1]), padding, stride)
    kernel_height, kernel_width = kernel_size
    batch_size, channels, _, _ = padded_input_array.shape

    output_array = engine.zeros((batch_size, channels, output_height, output_width))

    for row in range(output_height):
        for column in range(output_width):
            padded_input_slice = padded_input_array[:, :, row * stride:row * stride + kernel_height, column * stride:column * stride + kernel_width]
            output_array[:, :, row, column] = engine.mean(padded_input_slice, axis=(2, 3))

    return _create_tensor(
        inp,
        data=output_array,
        func=wrapped_partial(avg_pool_backward, inp=inp, kernel_size=kernel_size, stride=stride, padding=padding)
    )


def rnn_relu(inp, hx, all_weights, bias, num_layers):
    _check_tensors(inp)

    out_tensor = zeros((inp.size(0), inp.size(1), hx.size(2)))

    intermediate_values = {}
    for time in range(inp.size(1)):
        if time not in intermediate_values:
            intermediate_values[time] = {}

        out = inp[:, time, :]
        for layer in range(num_layers):
            if layer not in intermediate_values[time]:
                intermediate_values[time][layer] = {}

            if bias:
                w_ih, w_hh, b_ih, b_hh = all_weights[layer]
            else:
                w_ih, w_hh = all_weights[layer]
                b_ih, b_hh = None, None

            h = hx[layer]

            intermediate_values[time][layer]['input'] = out
            out = dense(out, w_ih, b_ih) + dense(h, w_hh, b_hh)
            intermediate_values[time][layer]['relu_input'] = out
            out = relu(out)
            intermediate_values[time][layer]['prev_h'] = h
            intermediate_values[time][layer]['current_h'] = out
            hx[layer] = out

        out_tensor[:, time, :] = out

    from functools import reduce
    out_tensor = _create_tensor(
        inp,
        *reduce(lambda x, y: x + y, all_weights),
        data=out_tensor.data,
        func=wrapped_partial(rnn_relu_backward, inp=inp, all_weights=all_weights, num_layers=num_layers, intermediate_values=intermediate_values)
    )
    return out_tensor, hx


def rnn_tanh(inp, hx, all_weights, bias, num_layers):
    # inp.shape = b, t, f
    # hx.shape = n, b, h

    out_tensor = zeros((inp.size(0), inp.size(1), hx.size(2)))
    for time in range(inp.size(1)):
        out = inp[:, time, :]
        for layer in range(num_layers):
            if bias:
                w_ih, w_hh, b_ih, b_hh = all_weights[layer]
            else:
                w_ih, w_hh = all_weights[layer]
                b_ih, b_hh = None, None

            h = hx[layer]

            out._data = tanh(dense(out, w_ih, b_ih) + dense(h, w_hh, b_hh)).data

            h._data = out.data

            hx[layer] = h.data  # need to stack, instead of assigning

        out_tensor[:, time, :] = out.data

    out_tensor = _create_tensor(
        inp,
        data=out_tensor.data,
        func=None
    )
    return out_tensor, hx


def lstm(inp, hx, all_weights, bias, num_layers):
    # inp.shape = b, t, f
    # hx.shape = n, b, h
    # cx.shape = n, b, h

    hx, cx = hx
    out_tensor = zeros((inp.size(0), inp.size(1), hx.size(2)))
    for time in range(inp.size(1)):
        out = inp[:, time, :]
        for layer in range(num_layers):
            if bias:
                w_ih, w_hh, b_ih, b_hh = all_weights[layer]
            else:
                w_ih, w_hh = all_weights[layer]
                b_ih, b_hh = None, None

            h = hx[layer]
            c = cx[layer]

            gates = dense(out, w_ih, b_ih) + dense(h, w_hh, b_hh)

            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = sigmoid(ingate)
            forgetgate = sigmoid(forgetgate)
            cellgate = tanh(cellgate)
            outgate = sigmoid(outgate)

            c._data = ((forgetgate * c) + (ingate * cellgate)).data
            out._data = (outgate * tanh(c)).data

            h._data = out.data

            cx[layer] = c.data
            hx[layer] = h.data  # need to stack, instead of assigning

        out_tensor[:, time, :] = out.data

    out_tensor = _create_tensor(
        inp,
        data=out_tensor.data,
        func=None
    )
    return out_tensor, (hx, cx)


def gru(inp, hx, all_weights, bias, num_layers):
    # inp.shape = b, t, f
    # hx.shape = n, b, h

    out_tensor = zeros((inp.size(0), inp.size(1), hx.size(2)))
    for time in range(inp.size(1)):
        out = inp[:, time, :]
        for layer in range(num_layers):
            if bias:
                w_ih, w_hh, b_ih, b_hh = all_weights[layer]
            else:
                w_ih, w_hh = all_weights[layer]
                b_ih, b_hh = None, None

            h = hx[layer]

            gi = dense(out, w_ih, b_ih)
            gh = dense(h, w_hh, b_hh)
            i_r, i_i, i_n = gi.chunk(3, 1)
            h_r, h_i, h_n = gh.chunk(3, 1)

            resetgate = sigmoid(i_r + h_r)
            inputgate = sigmoid(i_i + h_i)
            newgate = tanh(i_n + resetgate * h_n)
            out._data = (newgate + inputgate * (h - newgate)).data

            hx[layer] = out  # need to stack, instead of assigning

        out_tensor[:, time, :] = out

    out_tensor = _create_tensor(
        inp,
        data=out_tensor.data,
        func=None
    )
    return out_tensor, hx


def rnn_relu_cell(inp, h, w_ih, w_hh, b_ih=None, b_hh=None):
    return relu(dense(inp, w_ih, b_ih) + dense(h, w_hh, b_hh))


def rnn_tanh_cell(inp, h, w_ih, w_hh, b_ih=None, b_hh=None):
    return tanh(dense(inp, w_ih, b_ih) + dense(h, w_hh, b_hh))


def lstm_cell(inp, h, w_ih, w_hh, b_ih=None, b_hh=None):
    hx, cx = h
    gates = dense(inp, w_ih, b_ih) + dense(hx, w_hh, b_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = sigmoid(ingate)
    forgetgate = sigmoid(forgetgate)
    cellgate = tanh(cellgate)
    outgate = sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * tanh(cy)

    return hy, cy


def gru_cell(inp, h, w_ih, w_hh, b_ih=None, b_hh=None):
    gi = dense(inp, w_ih, b_ih)
    gh = dense(h, w_hh, b_hh)
    i_r, i_i, i_n = gi.chunk(3, 1)
    h_r, h_i, h_n = gh.chunk(3, 1)

    resetgate = sigmoid(i_r + h_r)
    inputgate = sigmoid(i_i + h_i)
    newgate = tanh(i_n + resetgate * h_n)
    hy = newgate + inputgate * (h - newgate)

    return hy


def adam(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, amsgrad, beta1, beta2, lr, weight_decay, eps):
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
            avg = engine.sqrt(square_avg - (grad_avg * grad_avg)) + eps
        else:
            avg = engine.sqrt(square_avg) + eps

        if momentum > 0:
            buf = momentum_buffers[i]
            # check if this is true
            buf._data = buf.data * momentum
            buf._data = buf.data + grad.data / avg
            # buf._data = buf.data * momentum + grad.data / avg
            param._data = param.data - lr * buf.data
        else:
            param._data = param.data - lr * grad.data / avg
