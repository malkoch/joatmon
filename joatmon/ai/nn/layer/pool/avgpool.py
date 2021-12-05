from ... import functional as f

__all__ = ['AvgPool']


class AvgPool:
    def __init__(self, kernel_size, stride, padding):
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding

    def forward(self, inp):
        return f.avg_pool(
            inp=inp,
            kernel_size=self._kernel_size,
            stride=self._stride,
            padding=self._padding
        )
