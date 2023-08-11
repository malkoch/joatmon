from joatmon.nn import functional as f
from joatmon.nn.core import Module

__all__ = ['Sigmoid']


class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, inp):
        return f.sigmoid(
            inp=inp
        )
