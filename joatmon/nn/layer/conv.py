import numpy as np

from joatmon.nn import functional as f
from joatmon.nn.core import (
    Module,
    Parameter,
    Tensor
)

__all__ = ['Conv', 'ConvTranspose']


class Conv(Module):
    """
    Applies a 2D convolution over an input signal composed of several input planes.

    # Arguments
        in_features (int): Number of channels in the input image.
        out_features (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0

    # Attributes
        weight (Tensor): The learnable weights of the module of shape (out_features, in_features, kernel_size).
        bias (Tensor): The learnable bias of the module of shape (out_features).
    """

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
        """
        Defines the computation performed at every call.

        # Arguments
            inp (Tensor): The input tensor.

        # Returns
            Tensor: The output tensor after applying 2D convolution.
        """
        return f.conv(
            inp=inp,
            weight=self.weight,
            bias=self.bias,
            stride=self._stride,
            padding=self._padding,
        )


class ConvTranspose(Module):
    """
    Applies a 2D transposed convolution operator over an input image composed of several input planes.

    # Arguments
        in_features (int): Number of channels in the input image.
        out_features (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0

    # Attributes
        weight (Tensor): The learnable weights of the module of shape (in_features, out_features, kernel_size).
        bias (Tensor): The learnable bias of the module of shape (out_features).
    """

    def __init__(self, in_features, out_features, kernel_size, stride=1, padding=0):
        super(ConvTranspose, self).__init__()

        self._in_features = in_features
        self._out_features = out_features
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding

        self.weight = Parameter(Tensor.from_array(np.ones((in_features, out_features, *kernel_size))))
        self.bias = Parameter(Tensor.from_array(np.ones((out_features,))))

    def forward(self, inp):
        """
        Defines the computation performed at every call.

        # Arguments
            inp (Tensor): The input tensor.

        # Returns
            Tensor: The output tensor after applying 2D transposed convolution.
        """
        return f.conv_transpose(
            inp=inp,
            weight=self.weight,
            bias=self.bias,
            stride=self._stride,
            padding=self._padding,
        )
