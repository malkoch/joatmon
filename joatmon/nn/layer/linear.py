import math

import numpy as np

from joatmon.nn import (
    functional as f,
    init
)
from joatmon.nn.core import (
    Module,
    Parameter,
    Tensor
)

__all__ = ['Linear']


class Linear(Module):
    """
    Applies a linear transformation to the incoming data: y = xA^T + b

    # Arguments
        in_features (int): size of each input sample
        out_features (int): size of each output sample
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Default: True

    # Attributes
        weight (Tensor): the learnable weights of the module of shape (out_features x in_features)
        bias (Tensor):   the learnable bias of the module of shape (out_features)
    """

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(Tensor.from_array(data=np.ones((self.out_features, self.in_features))))
        if bias:
            self.bias = Parameter(Tensor.from_array(data=np.zeros((self.out_features,))))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Resets the parameters (weight, bias) to their initial values.
        """
        init.kaiming_uniform(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform(self.bias, -bound, bound)

    def forward(self, inp):
        """
        Defines the computation performed at every call.

        # Arguments
            inp (Tensor): The input tensor.

        # Returns
            Tensor: The output tensor after applying linear transformation.
        """
        x = f.dense(inp=inp, weight=self.weight, bias=self.bias)
        return x

    def extra_repr(self) -> str:
        """
        Returns a string containing a brief description of the module.

        # Returns
            str: A string containing a brief description of the module.
        """
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
