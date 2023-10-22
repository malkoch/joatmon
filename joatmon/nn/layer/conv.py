import numpy as np

from joatmon.nn import functional as f
from joatmon.nn.core import (
    Module,
    Parameter,
    Tensor
)

__all__ = ['Conv', 'ConvTranspose']


class Conv(Module):
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

    def __init__(self, in_features, out_features, kernel_size, stride=1, padding=0):
        super(Conv, self).__init__()

        self._in_features = in_features
        self._out_features = out_features
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding

        self.weight = Parameter(Tensor.from_array(np.ones((out_features, in_features, *kernel_size))))
        self.bias = Parameter(Tensor.from_array(np.ones((out_features,))))

    def forward(self, inp):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return f.conv(
            inp=inp,
            weight=self.weight,
            bias=self.bias,
            stride=self._stride,
            padding=self._padding,
        )


class ConvTranspose(Module):
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

    def __init__(self, in_features, out_features, kernel_size, stride=1, padding=0):
        super(ConvTranspose, self).__init__()

        self._in_features = in_features
        self._out_features = out_features
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding

        self.weight = Parameter(Tensor.from_array(np.ones((out_features, in_features, *kernel_size))))
        self.bias = Parameter(Tensor.from_array(np.ones((out_features,))))

    def forward(self, inp):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return f.conv_transpose(
            inp=inp,
            weight=self.weight,
            bias=self.bias,
            stride=self._stride,
            padding=self._padding,
        )
