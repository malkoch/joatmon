import numpy as np

from joatmon.nn.core import Loss

__all__ = ['CCELoss']


class CCELoss(Loss):
    """
    Implements the Categorical Cross-Entropy (CCE) loss function.

    CCE is a loss function that is used in multi-class classification tasks.
    It is a measure of the dissimilarity between the predicted probability distribution and the true distribution.

    # Attributes
        _loss (np.array): The computed loss value.
    """

    def __init__(self):
        super(CCELoss, self).__init__()

        self._loss = None

    def forward(self, prediction, target) -> np.array:
        """
        Computes the CCE loss between the prediction and target.

        # Arguments
            prediction (np.array): The predicted probability distribution.
            target (np.array): The true distribution.

        # Returns
            np.array: The computed CCE loss.
        """
        # self._loss = -(target * prediction.log() + (1 - target) * (1 - prediction).log()).summation()
        self._loss = -((target * (prediction + 1e-100).log()).summation()) / np.prod(target.shape)
        return self._loss
