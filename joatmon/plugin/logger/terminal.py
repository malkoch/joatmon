import json

from joatmon.plugin.logger.core import LoggerPlugin


class TerminalLogger(LoggerPlugin):
    def __init__(self, level: str):
        super(TerminalLogger, self).__init__(level)

    async def _write(self, log: dict):
        print(json.dumps(log))
