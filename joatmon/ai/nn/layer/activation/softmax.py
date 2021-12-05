from ... import functional as f
from ...core import Module

__all__ = ['Softmax']


class Softmax(Module):
    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, inp):
        return f.softmax(
            inp=inp
        )
