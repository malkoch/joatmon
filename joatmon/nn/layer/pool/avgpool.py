from joatmon.nn import functional as f

__all__ = ['AvgPool']


class AvgPool:
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

    def __init__(self, kernel_size, stride, padding):
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding

    def forward(self, inp):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return f.avg_pool(inp=inp, kernel_size=self._kernel_size, stride=self._stride, padding=self._padding)
