import numpy as np

from joatmon.nn import functional
from joatmon.nn.core import Loss

__all__ = ['HuberLoss']


class HuberLoss(Loss):
    """
    Implements the Huber loss function.

    Huber loss is less sensitive to outliers in data than mean squared error.
    It's quadratic for small values of the input and linear for large values.

    # Attributes
        delta (float): The point where the Huber loss function changes from a quadratic to linear.
        _loss (np.array): The computed loss value.
    """

    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()

        self.delta = delta

        self._loss = None

    def forward(self, prediction, target) -> np.array:
        """
        Initializes the HuberLoss class.

        # Arguments
            delta (float, optional): The point where the Huber loss function changes from a quadratic to linear. Default is 1.0.
        """

        abs_diff = functional.absolute(target - prediction)
        quadratic = 0.5 * (abs_diff ** 2)
        linear = self.delta * (abs_diff - 0.5 * self.delta)
        loss = functional.where(abs_diff < self.delta, quadratic, linear)
        self._loss = functional.mean(loss)

        return self._loss
