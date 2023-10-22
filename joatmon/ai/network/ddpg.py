__all__ = ['DDPGActor', 'DDPGCritic']

from joatmon.array import functional
from joatmon.nn import (
    Module,
    Sequential
)
from joatmon.nn.layer.activation.relu import ReLU
from joatmon.nn.layer.activation.tanh import Tanh
from joatmon.nn.layer.batchnorm import BatchNorm
from joatmon.nn.layer.conv import Conv
from joatmon.nn.layer.linear import Linear


class DDPGActor(Module):
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

    def __init__(self, in_features, out_features):
        super(DDPGActor, self).__init__()

        self.extractor = Sequential(
            Conv(in_features=in_features, out_features=32, kernel_size=(8, 8), stride=(4, 4)),
            BatchNorm(features=32),
            ReLU(),
            Conv(in_features=32, out_features=64, kernel_size=(4, 4), stride=(2, 2)),
            BatchNorm(features=64),
            ReLU(),
            Conv(in_features=64, out_features=64, kernel_size=(3, 3), stride=(1, 1)),
            BatchNorm(features=64),
            ReLU(),
        )

        self.predictor = Sequential(
            Linear(in_features=7 * 7 * 64, out_features=200),
            BatchNorm(features=200),
            ReLU(),
            Linear(in_features=200, out_features=200),
            BatchNorm(features=200),
            ReLU(),
            Linear(in_features=200, out_features=out_features),
            Tanh(),
        )

    def forward(self, x):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        x = self.extractor(x)
        return self.predictor(x.view(x.size(0), -1))


class DDPGCritic(Module):
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

    def __init__(self, in_features, out_features):
        super(DDPGCritic, self).__init__()

        self.extractor = Sequential(
            Conv(in_features=in_features, out_features=32, kernel_size=(8, 8), stride=(4, 4)),
            BatchNorm(features=32),
            ReLU(),
            Conv(in_features=32, out_features=64, kernel_size=(4, 4), stride=(2, 2)),
            BatchNorm(features=64),
            ReLU(),
            Conv(in_features=64, out_features=64, kernel_size=(3, 3), stride=(1, 1)),
            BatchNorm(features=64),
            ReLU(),
        )

        self.relu = ReLU()

        self.linear1 = Linear(in_features=7 * 7 * 64, out_features=200)
        self.linear2 = Linear(in_features=200 + out_features, out_features=200)
        self.linear3 = Linear(in_features=200, out_features=1)

        self.bn1 = BatchNorm(features=200)
        self.bn2 = BatchNorm(features=200)

    def forward(self, x, y):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        x = self.extractor(x)

        x = self.relu(self.bn1(self.linear1(x.view(x.size(0), -1))))
        x = functional.concatenate([x, y], axis=1)
        x = self.relu(self.bn2(self.linear2(x)))
        return self.linear3(x)
