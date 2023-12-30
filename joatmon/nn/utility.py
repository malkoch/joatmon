import warnings
from typing import Any

EPOCH_DEPRECATION_WARNING = (
    'The epoch parameter in `scheduler.step()` was not necessary and is being '
    'deprecated where possible. Please use `scheduler.step()` to step the '
    'scheduler. During the deprecation, if epoch is different from None, the '
    'closed form is used instead of the new chainable form, where available. '
)

SAVE_STATE_WARNING = 'Please also save or load the state of the optimizer when saving or loading the scheduler.'


def _calculate_input_dims(output_shape, kernel_shape, padding, stride):  # instead of this, use is_transpose parameter and use function below
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    batch_size, _, output_height, output_width = output_shape
    _, out_filter_number, filter_height, filter_width = kernel_shape  # need to check

    input_height = (output_height - 1) * stride - 2 * padding[0] + (filter_height - 1) + 1
    input_width = (output_width - 1) * stride - 2 * padding[1] + (filter_width - 1) + 1
    return batch_size, out_filter_number, input_height, input_width


def _calculate_output_dims(input_shape, kernel_shape, padding, stride):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    batch_size, _, input_height, input_width = input_shape
    out_filter_number, _, filter_height, filter_width = kernel_shape

    output_height = (input_height + padding[0] * 2 - filter_height) // stride + 1
    output_width = (input_width + padding[1] * 2 - filter_width) // stride + 1
    return batch_size, out_filter_number, output_height, output_width


def _forward_unimplemented(self, *inp: Any) -> None:
    r"""Defines the computation performed at every call.

    Should be overridden by all subclasses.

    .. note::
        Although the recipe for forward pass needs to be defined within
        this function, one should call the :class:`Module` instance afterwards
        instead of this since the former takes care of running the
        registered hooks while the latter silently ignores them.
    """
    raise NotImplementedError


def typename(o):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    if type(o) == 'Tensor':
        return o.dtype

    module = ''
    if (
            hasattr(o, '__module__')
            and o.__module__ != 'builtins'
            and o.__module__ != '__builtin__'
            and o.__module__ is not None
    ):
        module = o.__module__ + '.'

    if hasattr(o, '__qualname__'):
        class_name = o.__qualname__
    elif hasattr(o, '__name__'):
        class_name = o.__name__
    else:
        class_name = o.__class__.__name__

    return module + class_name


def indent(string, spaces):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    s = string.split('\n')
    if len(s) == 1:
        return string
    first = s.pop(0)
    s = [(spaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


def get_enum(reduction):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    if reduction == 'none':
        ret = 0
    elif reduction == 'mean':
        ret = 1
    elif reduction == 'elementwise_mean':
        warnings.warn("reduction='elementwise_mean' is deprecated, please use reduction='mean' instead.")
        ret = 1
    elif reduction == 'sum':
        ret = 2
    else:
        ret = -1  # TODO: remove once JIT exceptions support control flow
        raise ValueError('{} is not a valid value for reduction'.format(reduction))
    return ret


def legacy_get_string(size_average, reduce, emit_warning=True):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    warning = "size_average and reduce args will be deprecated, please use reduction='{}' instead."

    if size_average is None:
        size_average = True
    if reduce is None:
        reduce = True

    if size_average and reduce:
        ret = 'mean'
    elif reduce:
        ret = 'sum'
    else:
        ret = 'none'
    if emit_warning:
        warnings.warn(warning.format(ret))
    return ret


def legacy_get_enum(size_average, reduce, emit_warning=True):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    return get_enum(legacy_get_string(size_average, reduce, emit_warning))


class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""

    def __repr__(self):
        return '<required parameter>'


required = _RequiredParameter()