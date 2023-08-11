import numpy as np

from joatmon.nn.core import Loss

__all__ = ['MAELoss']


class MAELoss(Loss):
    def __init__(self):
        super(MAELoss, self).__init__()

        self._loss = None

    def forward(self, prediction, target) -> np.array:
        self._loss = ((prediction - target).absolute() / np.prod(target.shape)).summation()
        return self._loss
