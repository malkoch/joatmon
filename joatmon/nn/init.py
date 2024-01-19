import math
import warnings


def _calculate_fan_in_and_fan_out(param):
    """
    Calculates the fan-in and fan-out of a tensor.

    Fan-in and fan-out can be thought of as the number of input and output units, respectively, in a weight tensor.

    Args:
        param (Tensor): Weight tensor

    Returns:
        tuple: A tuple containing the fan-in and fan-out of the weight tensor.
    """
    dimensions = param.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = param.size(1)
    num_output_fmaps = param.size(0)
    receptive_field_size = 1
    if param.dim() > 2:
        for s in param.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def _calculate_correct_fan(param, mode):
    """
    Calculates the correct fan value based on the mode.

    Args:
        param (Tensor): Weight tensor
        mode (str): Either 'fan_in' or 'fan_out'. Choosing 'fan_in' preserves the magnitude of the variance of the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes in the backwards pass.

    Returns:
        int: The correct fan value.
    """
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError(f"Mode {mode} not supported, please use one of {valid_modes}")

    fan_in, fan_out = _calculate_fan_in_and_fan_out(param)
    return fan_in if mode == 'fan_in' else fan_out


def calculate_gain(nonlinearity, param=None):
    """
    Returns the recommended gain value for the given nonlinearity function.

    The values are as follows:
    ============ ==========================================
    nonlinearity gain
    ============ ==========================================
    linear       :math:`1`
    conv{1,2,3}d :math:`1`
    sigmoid      :math:`1`
    tanh         :math:`5 / 3`
    relu         :math:`\sqrt{2}`
    leaky_relu   :math:`\sqrt{2 / (1 + negative\_slope^2)}`
    selu         :math:`3 / 4`
    ============ ==========================================

    Args:
        nonlinearity (str): The non-linear function (`nn.functional` name).
        param (float, optional): Optional parameter for the non-linear function.

    Returns:
        float: The recommended gain value.
    """
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            negative_slope = param
        else:
            raise ValueError(f"negative_slope {param} not a valid number")
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    elif nonlinearity == 'selu':
        return 3.0 / 4
    else:
        raise ValueError(f"Unsupported nonlinearity {nonlinearity}")


def normal(param, loc=0.0, scale=1.0):
    """
    Fills the input Tensor with values drawn from a normal distribution.

    Args:
        param (Tensor): an n-dimensional Tensor
        loc (float, optional): the mean of the normal distribution
        scale (float, optional): the standard deviation of the normal distribution
    """
    import numpy as np

    param._data = np.random.normal(loc, scale, size=param.shape)


def uniform(param, low=-1.0, high=1.0):
    """
    Fills the input Tensor with values drawn from a uniform distribution.

    Args:
        param (Tensor): an n-dimensional Tensor
        low (float, optional): the lower bound of the uniform distribution
        high (float, optional): the upper bound of the uniform distribution
    """
    import numpy as np

    param._data = np.random.uniform(low, high, size=param.shape)


def zeros(param):
    """
    Fills the input Tensor with zeros.

    Args:
        param (Tensor): an n-dimensional Tensor
    """
    import numpy as np

    param._data = np.zeros_like(param)


def ones(param):
    """
    Fills the input Tensor with ones.

    Args:
        param (Tensor): an n-dimensional Tensor
    """
    import numpy as np

    param._data = np.zeros_like(param)


def xavier_uniform(param, gain=1.):
    """
    Fills the input Tensor with values according to the method described in "Understanding the difficulty of training deep feedforward neural networks" - Glorot, X. & Bengio, Y. (2010), using a uniform distribution.

    Args:
        param (Tensor): an n-dimensional Tensor
        gain (float, optional): an optional scaling factor
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(param)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    uniform(param, -a, a)


def xavier_normal(param, gain=1.):
    """
    Fills the input Tensor with values according to the method described in "Understanding the difficulty of training deep feedforward neural networks" - Glorot, X. & Bengio, Y. (2010), using a normal distribution.

    Args:
        param (Tensor): an n-dimensional Tensor
        gain (float, optional): an optional scaling factor
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(param)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    normal(param, 0, std)


def kaiming_uniform(param, a: float = 0, mode: str = 'fan_in', nonlinearity: str = 'leaky_relu'):
    """
    Fills the input Tensor with values according to the method described in "Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification" - He, K. et al. (2015), using a uniform distribution.

    Args:
        param (Tensor): an n-dimensional Tensor
        a (float, optional): the negative slope of the rectifier used after this layer
        mode (str, optional): either 'fan_in' (default) or 'fan_out'. Choosing 'fan_in' preserves the magnitude of the variance of the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes in the backwards pass.
        nonlinearity (str, optional): the non-linear function (`nn.functional` name)
    """
    if 0 in param.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return
    fan = _calculate_correct_fan(param, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    uniform(param, -bound, bound)


def kaiming_normal(param, a: float = 0, mode: str = 'fan_in', nonlinearity: str = 'leaky_relu'):
    """
    Fills the input Tensor with values according to the method described in "Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification" - He, K. et al. (2015), using a normal distribution.

    Args:
        param (Tensor): an n-dimensional Tensor
        a (float, optional): the negative slope of the rectifier used after this layer
        mode (str, optional): either 'fan_in' (default) or 'fan_out'. Choosing 'fan_in' preserves the magnitude of the variance of the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes in the backwards pass.
        nonlinearity (str, optional): the non-linear function (`nn.functional` name)
    """
    if 0 in param.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return
    fan = _calculate_correct_fan(param, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    normal(param, 0, std)
