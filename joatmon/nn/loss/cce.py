import numpy as np

from joatmon.nn.core import Loss

__all__ = ['CCELoss']


class CCELoss(Loss):
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

    def __init__(self):
        super(CCELoss, self).__init__()

        self._loss = None

    def forward(self, prediction, target) -> np.array:
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        # self._loss = -(target * prediction.log() + (1 - target) * (1 - prediction).log()).summation()
        self._loss = -((target * (prediction + 1e-100).log()).summation()) / np.prod(target.shape)
        return self._loss
