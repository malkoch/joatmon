__all__ = ['DDPGActor', 'DDPGCritic']

from joatmon.nn import (
    Module,
    Sequential, functional
)
from joatmon.nn.layer.activation.relu import ReLU
from joatmon.nn.layer.activation.tanh import Tanh
from joatmon.nn.layer.batchnorm import BatchNorm
from joatmon.nn.layer.conv import Conv
from joatmon.nn.layer.linear import Linear


class DDPGActor(Module):
    """
    Deep Deterministic Policy Gradient Actor Model.

    This class is used to create the actor model for the DDPG algorithm.
    The actor model is responsible for selecting actions based on the current state of the environment.

    Attributes:
        extractor (Sequential): A sequence of convolutional layers used for feature extraction.
        predictor (Sequential): A sequence of linear layers used for action prediction.

    Args:
        in_features (int): The number of input features.
        out_features (int): The number of output features (actions).
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
        Forward pass through the actor model.

        Args:
            x (Tensor): The input tensor representing the current state of the environment.

        Returns:
            Tensor: The output tensor representing the selected action.
        """
        x = self.extractor(x)
        return self.predictor(x.view(x.size(0), -1))


class DDPGCritic(Module):
    """
    Deep Deterministic Policy Gradient Critic Model.

    This class is used to create the critic model for the DDPG algorithm.
    The critic model is responsible for evaluating the value of the selected action.

    Attributes:
        extractor (Sequential): A sequence of convolutional layers used for feature extraction.
        relu (ReLU): The ReLU activation function.
        linear1 (Linear): The first linear layer.
        linear2 (Linear): The second linear layer.
        linear3 (Linear): The third linear layer.
        bn1 (BatchNorm): The first batch normalization layer.
        bn2 (BatchNorm): The second batch normalization layer.

    Args:
        in_features (int): The number of input features.
        out_features (int): The number of output features (actions).
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
        Forward pass through the critic model.

        Args:
            x (Tensor): The input tensor representing the current state of the environment.
            y (Tensor): The input tensor representing the selected action.

        Returns:
            Tensor: The output tensor representing the value of the selected action.
        """
        x = self.extractor(x)

        x = self.relu(self.bn1(self.linear1(x.view(x.size(0), -1))))
        x = functional.concat([x, y], axis=1)
        x = self.relu(self.bn2(self.linear2(x)))
        return self.linear3(x)
