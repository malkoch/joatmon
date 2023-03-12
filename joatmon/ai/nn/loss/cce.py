import numpy as np

from joatmon.ai.nn.core import Loss

__all__ = ['CCELoss']


class CCELoss(Loss):
    def __init__(self):
        super(CCELoss, self).__init__()

        self._loss = None

    def forward(self, prediction, target) -> np.array:
        self._loss = -(target * np.log(prediction.data) + (1 - target) * np.log(1 - prediction.data)).summation()
        return self._loss
