from typing import Union

from joatmon import context
from joatmon.orm.enum import Enum
from joatmon.plugin.core import Plugin


class LogLevel(Enum):
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

    NotSet = 1 << 0
    Debug = 1 << 1
    Info = 1 << 2
    Warning = 1 << 3
    Error = 1 << 4
    Critical = 1 << 5


class LoggerPlugin(Plugin):
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

    def __init__(self, level, language, ip):
        if isinstance(level, LogLevel):
            level = level.name
        self._level = LoggerPlugin._get_level(level)

        self.language = language
        self.ip = ip

    @staticmethod
    def _get_level(level_str):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        _level_key = list(filter(lambda x: level_str.lower() == x.lower(), LogLevel.__dict__.keys()))
        if len(_level_key) == 1:
            _level_key = _level_key[0]
        else:
            _level_key = LogLevel.Info.name

        return LogLevel[_level_key]

    async def _write(self, log: dict):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        raise NotImplementedError

    async def log(self, log: dict, level: Union[LogLevel, str] = LogLevel.Debug):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        try:
            if isinstance(level, str):
                level = LoggerPlugin._get_level(level)

            if self._level.value <= level.value:
                log['level'] = level.name
                log['language'] = context.get_value(self.language).get()
                log['ip'] = context.get_value(self.ip).get()
                await self._write(log)
        except Exception as ex:
            print(str(ex))

    async def debug(self, log):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        await self.log(log=log, level=LogLevel.Debug)

    async def info(self, log):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        await self.log(log=log, level=LogLevel.Info)

    async def warning(self, log):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        await self.log(log=log, level=LogLevel.Warning)

    async def error(self, log):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        await self.log(log=log, level=LogLevel.Error)

    async def critical(self, log):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        await self.log(log=log, level=LogLevel.Critical)
