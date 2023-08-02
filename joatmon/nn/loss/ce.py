import numpy as np

from joatmon.nn.core import Loss

__all__ = ['CELoss']


class CELoss(Loss):
    def __init__(self):
        super(CELoss, self).__init__()

        self._loss = None

    def forward(self, prediction, target) -> np.array:
        self._loss = -(target * np.log(prediction.data) + (1 - target) * np.log(1 - prediction.data)).summation()
        return self._loss
