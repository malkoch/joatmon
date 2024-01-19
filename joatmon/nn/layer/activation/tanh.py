from joatmon.nn import functional as f
from joatmon.nn.core import Module

__all__ = ['Tanh']


class Tanh(Module):
    """
    Applies the Tanh (Hyperbolic Tangent) activation function to the input.

    This class does not require any arguments during initialization.
    """

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, inp):
        """
        Applies the Tanh activation function to the input.

        # Arguments
            inp (Tensor): The input tensor.

        # Returns
            Tensor: The output tensor with the same shape as the input.
        """
        return f.tanh(inp=inp)
