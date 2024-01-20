from typing import Union

from joatmon.core import context
from joatmon.orm.enum import Enum
from joatmon.plugin.core import Plugin


class LogLevel(Enum):
    """
    LogLevel is an enumeration that defines the different levels of logging.

    Attributes:
        NotSet (int): Level for not setting a logging level.
        Debug (int): Level for debug logging.
        Info (int): Level for information logging.
        Warning (int): Level for warning logging.
        Error (int): Level for error logging.
        Critical (int): Level for critical logging.
    """

    NotSet = 1 << 0
    Debug = 1 << 1
    Info = 1 << 2
    Warning = 1 << 3
    Error = 1 << 4
    Critical = 1 << 5


class LoggerPlugin(Plugin):
    """
    LoggerPlugin is a class that provides logging functionality.

    Attributes:
        _level (LogLevel): The level of logging.
        language (str): The language for logging.
        ip (str): The IP address for logging.
    """

    def __init__(self, level, language, ip):
        """
        Initialize LoggerPlugin with the given level, language, and IP.

        Args:
            level (LogLevel): The level of logging.
            language (str): The language for logging.
            ip (str): The IP address for logging.
        """
        if isinstance(level, LogLevel):
            level = level.name
        self._level = LoggerPlugin._get_level(level)

        self.language = language
        self.ip = ip

    @staticmethod
    def _get_level(level_str):
        """
        Get the LogLevel from a string.

        Args:
            level_str (str): The string representation of the LogLevel.

        Returns:
            LogLevel: The LogLevel corresponding to the given string.
        """
        _level_key = list(filter(lambda x: level_str.lower() == x.lower(), LogLevel.__dict__.keys()))
        if len(_level_key) == 1:
            _level_key = _level_key[0]
        else:
            _level_key = LogLevel.Info.name

        return LogLevel[_level_key]

    async def _write(self, log: dict):
        """
        Write a log to the logger.

        Args:
            log (dict): The log to be written.

        Raises:
            NotImplementedError: This method should be implemented in the child classes.
        """
        raise NotImplementedError

    async def log(self, log: dict, level: Union[LogLevel, str] = LogLevel.Debug):
        """
        Log a message at a specified level.

        Args:
            log (dict): The log to be written.
            level (Union[LogLevel, str]): The level at which to log the message.
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
        Log a debug message.

        Args:
            log (dict): The log to be written.
        """
        await self.log(log=log, level=LogLevel.Debug)

    async def info(self, log):
        """
        Log an info message.

        Args:
            log (dict): The log to be written.
        """
        await self.log(log=log, level=LogLevel.Info)

    async def warning(self, log):
        """
        Log a warning message.

        Args:
            log (dict): The log to be written.
        """
        await self.log(log=log, level=LogLevel.Warning)

    async def error(self, log):
        """
        Log an error message.

        Args:
            log (dict): The log to be written.
        """
        await self.log(log=log, level=LogLevel.Error)

    async def critical(self, log):
        """
        Log a critical message.

        Args:
            log (dict): The log to be written.
        """
        await self.log(log=log, level=LogLevel.Critical)
