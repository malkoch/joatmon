from joatmon.nn import functional as f
from joatmon.nn.core import (
    Module
)

__all__ = ['Upsample']


class Upsample(Module):
    """
    Upsamples an input.

    The input data is assumed to be of the form `minibatch x channels x [optional depth] x [optional height] x width`.
    The modes available for upsampling are: `nearest`.

    # Arguments
        scale_factor (int or tuple, optional): multiplier for spatial size. Has to match input size if it is a tuple.
        mode (str, optional): the upsampling algorithm: one of `nearest`. Default: `nearest`

    # Attributes
        _scale_factor (int or tuple): multiplier for spatial size. Has to match input size if it is a tuple.
        _mode (str): the upsampling algorithm: one of `nearest`.
    """

    def __init__(self, scale_factor=None, mode='nearest'):
        super(Upsample, self).__init__()

        self._scale_factor = scale_factor
        self._mode = mode

    def forward(self, inp):
        """
        Defines the computation performed at every call.

        # Arguments
            inp (Tensor): The input tensor.

        # Returns
            Tensor: The output tensor after applying upsampling.
        """
        if self._mode not in ('bilinear',):
            raise ValueError(f'{self._mode} is not supported')

        if self._mode == 'bilinear':
            return f.bilinear_interpolation(
                inp=inp,
                scale_factor=self._scale_factor
            )
