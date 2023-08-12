from joatmon import context
from joatmon.plugin.logger.core import LoggerPlugin


class DatabaseLogger(LoggerPlugin):
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

    def __init__(self, level: str, database: str, cls, language, ip):
        super(DatabaseLogger, self).__init__(level, language, ip)

        self.database = database
        self.cls = cls

    async def _write(self, log: dict):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        database = context.get_value(self.database)
        await database.insert(self.cls, log)
