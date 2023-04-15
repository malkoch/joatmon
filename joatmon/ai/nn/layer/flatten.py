from joatmon.ai.nn import functional as f
from joatmon.ai.nn.core import Module

__all__ = ['Flatten']


class Flatten(Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, inp):
        return f.view(
            inp=inp
        )
