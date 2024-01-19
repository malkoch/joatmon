import numpy as np

from joatmon.nn.core import Loss

__all__ = ['CELoss']


class CELoss(Loss):
    """
    Implements the Cross-Entropy (CE) loss function.

    CE is a loss function that is used in binary classification tasks.
    It is a measure of the dissimilarity between the predicted probability and the true label.

    # Attributes
        _loss (np.array): The computed loss value.
    """

    def __init__(self):
        super(CELoss, self).__init__()

        self._loss = None

    def forward(self, prediction, target) -> np.array:
        """
        Computes the CE loss between the prediction and target.

        # Arguments
            prediction (np.array): The predicted probability.
            target (np.array): The true label.

        # Returns
            np.array: The computed CE loss.
        """
        self._loss = -(target * np.log(prediction.data) + (1 - target) * np.log(1 - prediction.data)).summation()
        return self._loss
