from joatmon.nn import functional as f
from joatmon.nn.core import (
    Module
)

__all__ = ['Upsample']


class Upsample(Module):
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __init__(self, scale_factor=None, mode='nearest'):
        super(Upsample, self).__init__()

        self._scale_factor = scale_factor
        self._mode = mode

    def forward(self, inp):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if self._mode not in ('bilinear',):
            raise ValueError(f'{self._mode} is not supported')

        if self._mode == 'bilinear':
            return f.bilinear_interpolation(
                inp=inp,
                scale_factor=self._scale_factor
            )
