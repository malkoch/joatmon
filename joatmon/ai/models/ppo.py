from joatmon.nn import (
    concat,
    Module,
    relu,
    Tensor
)
from joatmon.nn.layer.linear import Linear


class PPOActor(Module):
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
        super(PPOActor, self).__init__()

        self.hidden1 = Linear(in_features, 128)
        self.hidden2 = Linear(128, 128)
        self.out = Linear(128, out_features)

    def forward(self, state: Tensor) -> Tensor:
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        x = relu(self.hidden1(state))
        x = relu(self.hidden2(x))
        action = self.out(x).tanh()

        return action


class PPOCritic(Module):
    def __init__(self, in_features, out_features):
        super(PPOCritic, self).__init__()

        self.hidden1 = Linear(in_features + out_features, 128)
        self.hidden2 = Linear(128, 128)
        self.out = Linear(128, 1)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        x = concat([state, action], axis=-1)
        x = relu(self.hidden1(x))
        x = relu(self.hidden2(x))
        value = self.out(x)

        return value
