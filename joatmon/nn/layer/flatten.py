from joatmon.nn import functional as f
from joatmon.nn.core import Module

__all__ = ['Flatten']


class Flatten(Module):
    """
    Flattens the input. Does not affect the batch size.

    This layer is often used to flatten the output of a convolutional layer,
    which is multi-dimensional, into a one-dimensional tensor (or 'vector'),
    so that all the output units can be connected to all the input units in the next layer.

    # Attributes
        None
    """

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, inp):
        """
        Defines the computation performed at every call.

        # Arguments
            inp (Tensor): The input tensor.

        # Returns
            Tensor: The output tensor after flattening the input.
        """
        return f.view(inp=inp)
