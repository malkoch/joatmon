from joatmon.nn import (
    functional as f,
    Module
)

__all__ = ['Dropout']


class Dropout(Module):
    """
    Applies Dropout to the input.

    The Dropout layer randomly sets input units to 0 with a frequency of `keep_prob`
    at each step during training time, which helps prevent overfitting.
    Inputs not set to 0 are scaled up by 1/(1 - keep_prob) such that the sum over
    all inputs is unchanged.

    # Arguments
        keep_prob (float): float between 0 and 1. Fraction of the input units to drop.

    # Attributes
        _keep_prob (float): Fraction of the input units to drop.
    """

    def __init__(self, keep_prob=.5):
        super(Dropout, self).__init__()

        self._keep_prob = keep_prob

    def forward(self, inp):
        """
        Applies Dropout to the input.

        # Arguments
            inp (Tensor): The input tensor.

        # Returns
            Tensor: The output tensor after applying dropout.
        """
        if self.training:
            return f.dropout(inp=inp, keep_prob=self._keep_prob)
        else:
            return inp
