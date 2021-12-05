from .. import functional as f

__all__ = ['Dropout']


class Dropout:
    def __init__(self, keep_prob):
        self._keep_prob = keep_prob

    def forward(self, inp):
        return f.dropout(
            inp=inp,
            keep_prob=self._keep_prob
        )
