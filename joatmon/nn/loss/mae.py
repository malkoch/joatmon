import numpy as np

from joatmon.nn.core import Loss

__all__ = ['MAELoss']


class MAELoss(Loss):
    """
    Implements the Mean Absolute Error (MAE) loss function.

    MAE is a loss function used for regression models. It is the sum of the absolute differences between the true and predicted values.

    # Attributes
        _loss (np.array): The computed loss value.
    """

    def __init__(self):
        """
        Initializes the MAELoss class.
        """
        super(MAELoss, self).__init__()

        self._loss = None

    def forward(self, prediction, target) -> np.array:
        """
        Computes the MAE loss between the prediction and target.

        # Arguments
            prediction (np.array): The predicted values.
            target (np.array): The true values.

        # Returns
            np.array: The computed MAE loss.
        """
        self._loss = ((prediction - target).absolute() / np.prod(target.shape)).summation()
        return self._loss
