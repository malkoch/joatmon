import numpy as np

from .. import functional as f
from ..core import (
    Module,
    Tensor,
    Parameter
)

__all__ = ['BatchNorm']


class BatchNorm(Module):
    def __init__(self, features, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True, track_running_stats: bool = True):
        super(BatchNorm, self).__init__()

        self._features = features
        self._eps = eps
        self._momentum = momentum
        self._affine = affine
        self._track_running_stats = track_running_stats
        self._training = True

        if self._affine:
            self.weight = Parameter(Tensor.from_array(np.ones((features,))))
            self.bias = Parameter(Tensor.from_array(np.zeros((features,))))
        else:
            self.weight = None
            self.bias = None

        if self._track_running_stats:
            self.running_mean = Parameter(Tensor.from_array(np.zeros((features,))))
            self.running_var = Parameter(Tensor.from_array(np.ones((features,))))
            self.num_batches_tracked = 0
        else:
            self.running_mean = None
            self.running_var = None
            self.num_batches_tracked = None

        if self._track_running_stats:
            self.running_mean = Parameter(Tensor.from_array(np.zeros((features,))))
            self.running_var = Parameter(Tensor.from_array(np.ones((features,))))
            self.num_batches_tracked = 0

        self.reset_parameters()

    def reset_running_stats(self):
        if self._track_running_stats:
            self.running_mean = Parameter(Tensor.from_array(np.zeros((self._features,))))
            self.running_var = Parameter(Tensor.from_array(np.ones((self._features,))))
            self.num_batches_tracked = 0

    def reset_parameters(self):
        self.reset_running_stats()
        if self._affine:
            self.weight = Parameter(Tensor.from_array(np.ones((self._features,))))
            self.bias = Parameter(Tensor.from_array(np.zeros((self._features,))))

    def forward(self, inp):
        if self._momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self._momentum

        if self._training and self._track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self._momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self._momentum

        return f.batch_norm(
            inp=inp,
            weight=self.weight,
            bias=self.bias,
            running_mean=self.running_mean if not self._training or self._track_running_stats else None,
            running_var=self.running_var if not self._training or self._track_running_stats else None,
            momentum=exponential_average_factor,
            eps=self._eps,
            training=self._training or ((self.running_mean is None) and (self.running_var is None))
        )
