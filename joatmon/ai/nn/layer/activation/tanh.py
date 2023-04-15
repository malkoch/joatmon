from joatmon.ai.nn import functional as f
from joatmon.ai.nn.core import Module

__all__ = ['TanH']


class TanH(Module):
    def __init__(self):
        super(TanH, self).__init__()

    def forward(self, inp):
        return f.tanh(
            inp=inp
        )
