import numpy as np

from joatmon.nn.core import Loss

__all__ = ['MSELoss']


class MSELoss(Loss):
    def __init__(self):
        super(MSELoss, self).__init__()

        self._loss = None

    def forward(self, prediction, target) -> np.array:
        self._loss = (((prediction - target) ** 2) / np.prod(target.shape)).summation()
        return self._loss
