from joatmon.ai.nn import functional as f
from joatmon.ai.nn.core import Module

__all__ = ['ReLU']


class ReLU(Module):
    def __init__(self, alpha=0):
        super(ReLU, self).__init__()

        self.alpha = alpha

    def forward(self, inp):
        return f.relu(
            inp=inp,
            alpha=self.alpha
        )
