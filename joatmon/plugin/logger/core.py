from enum import Enum
from typing import Union

from joatmon.core import CoreException
from joatmon.plugin.core import Plugin


class LoggerException(CoreException):
    ...


class LogLevel(Enum):
    NotSet = 1 << 0
    Debug = 1 << 1
    Info = 1 << 2
    Warning = 1 << 3
    Error = 1 << 4
    Critical = 1 << 5


class Logger(Plugin):
    def __init__(self, alias: str, level: str):
        super(Logger, self).__init__(alias)

        self._level = Logger._get_level(level)

    @staticmethod
    def _get_level(level_str):
        _level_key = list(filter(lambda x: level_str.lower() == x.lower(), LogLevel.__dict__.keys()))
        if len(_level_key) == 1:
            _level_key = _level_key[0]
        else:
            raise ValueError(f'could not found the log level {level_str}')

        return LogLevel[_level_key]

    def _write(self, log: dict):
        raise NotImplementedError

    def log(self, log: dict, level: Union[LogLevel, str] = LogLevel.Debug):
        try:
            if isinstance(level, str):
                level = Logger._get_level(level)

            if self._level.value <= level.value:
                log['level'] = level.name
                self._write(log)
        except Exception as ex:
            print(str(ex))

    def debug(self, log):
        self.log(log=log, level=LogLevel.Debug)

    def info(self, log):
        self.log(log=log, level=LogLevel.Info)

    def warning(self, log):
        self.log(log=log, level=LogLevel.Warning)

    def error(self, log):
        self.log(log=log, level=LogLevel.Error)

    def critical(self, log):
        self.log(log=log, level=LogLevel.Critical)
