import numpy as np

from .. import functional as f
from ..core import (
    Module,
    Tensor,
    Parameter
)

__all__ = ['Conv']


class Conv(Module):
    def __init__(self, in_features, out_features, kernel_size, stride=1, padding=0):
        super(Conv, self).__init__()

        self._in_features = in_features
        self._out_features = out_features
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding

        self.weight = Parameter(Tensor.from_array(np.ones((out_features, in_features, *kernel_size))))
        self.bias = Parameter(Tensor.from_array(np.ones((out_features,))))

    def forward(self, inp):
        return f.conv(
            inp=inp,
            weight=self.weight,
            bias=self.bias,
            stride=self._stride,
            padding=self._padding,
        )
