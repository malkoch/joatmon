from joatmon.nn import functional as f
from joatmon.nn.core import Module

__all__ = ['ReLU']


class ReLU(Module):
    """
    Applies the ReLU (Rectified Linear Unit) activation function to the input.

    # Arguments
        alpha (float): Controls the slope for values less than zero. Default is 0.

    # Attributes
        alpha (float): Controls the slope for values less than zero.
    """

    def __init__(self, alpha=0):
        super(ReLU, self).__init__()

        self.alpha = alpha

    def forward(self, inp):
        """
        Applies the ReLU activation function to the input.

        # Arguments
            inp (Tensor): The input tensor.

        # Returns
            Tensor: The output tensor with the same shape as the input.
        """
        return f.relu(inp=inp, alpha=self.alpha)
