from ... import functional as f
from ...core import Module

__all__ = ['ReLU']


class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, inp):
        return f.relu(
            inp=inp
        )
