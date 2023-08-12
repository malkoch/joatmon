from joatmon import context


def register(cls, alias, *args, **kwargs):
    context.set_value(alias.replace('-', '_'), PluginProxy(cls, *args, **kwargs))


class Plugin:
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

    ...


class PluginProxy:
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

    def __init__(self, cls, *args, **kwargs):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

        self.instance = None

    def __getattr__(self, item):
        if self.instance is None:
            self.instance = self.cls(*self.args, **self.kwargs)
        return getattr(self.instance, item.replace('-', '_'))

    async def __aenter__(self):
        if self.instance is None:
            self.instance = self.cls(*self.args, **self.kwargs)
        return await self.instance.__aenter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.instance is None:
            self.instance = self.cls(*self.args, **self.kwargs)
        return await self.instance.__aexit__(exc_type, exc_val, exc_tb)
