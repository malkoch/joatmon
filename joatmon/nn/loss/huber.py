import numpy as np

from joatmon.nn import functional
from joatmon.nn.core import Loss

__all__ = ['HuberLoss']


class HuberLoss(Loss):
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

    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()

        self.delta = delta

        self._loss = None

    def forward(self, prediction, target) -> np.array:
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """

        abs_diff = functional.absolute(target - prediction)
        quadratic = 0.5 * (abs_diff ** 2)
        linear = self.delta * (abs_diff - 0.5 * self.delta)
        loss = functional.where(abs_diff < self.delta, quadratic, linear)
        self._loss = functional.mean(loss)

        return self._loss
