import datetime
import json
import os

from joatmon.plugin.logger.core import LoggerPlugin


class FileLogger(LoggerPlugin):
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

    def __init__(self, level: str, base_folder: str, language, ip):
        super(FileLogger, self).__init__(level, language, ip)

        self.folder = base_folder

    async def _write(self, log: dict):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        dt = datetime.datetime.now()
        folder = os.path.join(self.folder, dt.strftime(f'%Y{os.sep}%m{os.sep}'))
        if not os.path.exists(folder):
            os.makedirs(folder)
        file = os.path.join(folder, dt.strftime(f'%d') + '.log')
        with open(file, 'a') as f:
            f.write(json.dumps(log) + os.linesep)
