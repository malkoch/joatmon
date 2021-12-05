from ... import functional as f
from ...core import Module

__all__ = ['TanH']


class TanH(Module):
    def __init__(self):
        super(TanH, self).__init__()

    def forward(self, inp):
        return f.tanh(
            inp=inp
        )
