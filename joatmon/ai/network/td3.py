from joatmon.nn import (
    functional,
    Module,
    Tensor
)
from joatmon.nn.layer.linear import Linear


class TD3Actor(Module):
    """
    Twin Delayed Deep Deterministic Policy Gradient Actor Model.

    This class is used to create the actor model for the TD3 algorithm.
    The actor model is responsible for selecting actions based on the current state of the environment.

    Attributes:
        hidden1 (Linear): The first hidden layer.
        hidden2 (Linear): The second hidden layer.
        out (Linear): The output layer.

    Args:
        in_features (int): The number of input features.
        out_features (int): The number of output features (actions).
    """

    def __init__(self, in_features, out_features):
        super(TD3Actor, self).__init__()

        self.hidden1 = Linear(in_features, 128)
        self.hidden2 = Linear(128, 128)
        self.out = Linear(128, out_features)

    def forward(self, state: Tensor) -> Tensor:
        """
        Forward pass through the actor model.

        Accepts a state and returns the predicted action.

        Args:
            state (Tensor): The input tensor representing the current state of the environment.

        Returns:
            Tensor: The output tensor representing the predicted action.
        """
        x = functional.relu(self.hidden1(state))
        x = functional.relu(self.hidden2(x))
        action = self.out(x).tanh()

        return action


class TD3Critic(Module):
    """
    Twin Delayed Deep Deterministic Policy Gradient Critic Model.

    This class is used to create the critic model for the TD3 algorithm.
    The critic model is responsible for evaluating the value of the selected action.

    Attributes:
        hidden1 (Linear): The first hidden layer.
        hidden2 (Linear): The second hidden layer.
        out (Linear): The output layer.

    Args:
        in_features (int): The number of input features.
        out_features (int): The number of output features (actions).
    """

    def __init__(self, in_features, out_features):
        super(TD3Critic, self).__init__()

        self.hidden1 = Linear(in_features + out_features, 128)
        self.hidden2 = Linear(128, 128)
        self.out = Linear(128, 1)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        """
        Forward pass through the critic model.

        Accepts a state and action and returns the value of the selected action.

        Args:
            state (Tensor): The input tensor representing the current state of the environment.
            action (Tensor): The input tensor representing the selected action.

        Returns:
            Tensor: The output tensor representing the value of the selected action.
        """
        x = functional.concat([state, action], axis=-1)
        x = functional.relu(self.hidden1(x))
        x = functional.relu(self.hidden2(x))
        value = self.out(x)

        return value
