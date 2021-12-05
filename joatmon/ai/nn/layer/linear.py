import numpy as np

from .. import functional as f
from ..core import (
    Module,
    Tensor,
    Parameter
)

__all__ = ['Linear']


class Linear(Module):
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
        self.weight = Parameter(Tensor.from_array(data=np.ones((self.out_features, self.in_features))))
        if self.bias is not None:
            self.bias = Parameter(Tensor.from_array(data=np.zeros((self.out_features,))))

    def forward(self, inp):
        x = f.dense(
            inp=inp,
            weight=self.weight,
            bias=self.bias
        )
        return x

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias is not None)
