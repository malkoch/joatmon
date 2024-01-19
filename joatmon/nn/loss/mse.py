import numpy as np

from joatmon.nn.core import Loss

__all__ = ['MSELoss']


class MSELoss(Loss):
    """
    Implements the Mean Squared Error (MSE) loss function.

    MSE is a loss function used for regression models. It is the sum of the squared differences between the true and predicted values.

    # Attributes
        _loss (np.array): The computed loss value.
    """

    def __init__(self):
        """
        Initializes the MSELoss class.
        """
        super(MSELoss, self).__init__()

        self._loss = None

    def forward(self, prediction, target) -> np.array:
        """
        Computes the MSE loss between the prediction and target.

        # Arguments
            prediction (np.array): The predicted values.
            target (np.array): The true values.

        # Returns
            np.array: The computed MSE loss.
        """
        self._loss = (((prediction - target) ** 2) / np.prod(target.shape)).summation()
        return self._loss
