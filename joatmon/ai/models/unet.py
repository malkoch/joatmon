from joatmon.ai.models.core import CoreModel
from joatmon.ai.network import UNet
from joatmon.ai.utility import (
    load,
    save
)
from joatmon.nn import init
from joatmon.nn.layer.batchnorm import BatchNorm

from joatmon.nn.layer.conv import Conv
from joatmon.nn.layer.linear import Linear

from joatmon.nn.loss.huber import HuberLoss

from joatmon.nn.optimizer.adam import Adam

__all__ = ['UNetModel']


class UNetModel(CoreModel):
    """
    U Network

    # Arguments
        models (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __init__(self, lr=1e-3, channels=3, classes=10):
        super().__init__()

        self.net = UNet(channels, classes, False)

        self.lr = lr

        self.optimizer = Adam(list(self.net.parameters()), lr=self.lr)
        self.loss = HuberLoss()

    def load(self, path=''):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        load(self.net, path)

    def save(self, path=''):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        save(self.net, path)

    def initialize(self, w_init=None, b_init=None):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        for module in self.net.modules():
            if isinstance(
                    module, (Conv, BatchNorm, Linear)
            ):  # batch norm will be different
                if w_init:
                    init.kaiming_normal(module.weight)
                if b_init:  # bias init will be different
                    init.kaiming_normal(module.bias)

    def predict(self, state=None):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...

    def train(self, batch=None, update_target=False):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...

    def evaluate(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...
