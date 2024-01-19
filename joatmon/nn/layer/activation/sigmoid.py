from joatmon.nn import functional as f
from joatmon.nn.core import Module

__all__ = ['Sigmoid']


class Sigmoid(Module):
    """
    Applies the Sigmoid activation function to the input.

    This class does not require any arguments during initialization.
    """

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, inp):
        """
        Applies the Sigmoid activation function to the input.

        # Arguments
            inp (Tensor): The input tensor.

        # Returns
            Tensor: The output tensor with the same shape as the input.
        """
        return f.sigmoid(inp=inp)
