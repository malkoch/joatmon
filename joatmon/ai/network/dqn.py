from joatmon.nn import (
    Module,
    Sequential
)
from joatmon.nn.layer.activation.relu import ReLU
from joatmon.nn.layer.batchnorm import BatchNorm
from joatmon.nn.layer.conv import Conv
from joatmon.nn.layer.linear import Linear


class DQN(Module):
    """
    Deep Q-Network Model.

    This class is used to create the DQN model for the DQN algorithm.
    The DQN model is responsible for selecting actions based on the current state of the environment.

    Attributes:
        extractor (Sequential): A sequence of convolutional layers used for feature extraction.
        predictor (Sequential): A sequence of linear layers used for action prediction.

    Args:
        in_features (int): The number of input features.
        out_features (int): The number of output features (actions).
    """

    def __init__(self, in_features, out_features):
        super(DQN, self).__init__()

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
            Linear(in_features=7 * 7 * 64, out_features=512),
            BatchNorm(features=512),
            ReLU(),
            Linear(in_features=512, out_features=out_features),
        )

    def forward(self, x):
        """
        Forward pass through the DQN model.

        Accepts a state and returns the predicted action.

        Args:
            x (Tensor): The input tensor representing the current state of the environment.

        Returns:
            Tensor: The output tensor representing the predicted action.
        """
        x = self.extractor(x)
        return self.predictor(x.view(x.size(0), -1))
