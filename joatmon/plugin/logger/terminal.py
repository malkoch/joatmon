import json

from joatmon.plugin.logger.core import LoggerPlugin


class TerminalLogger(LoggerPlugin):
    def __init__(self, level: str, language, ip):
        super(TerminalLogger, self).__init__(level, language, ip)

    async def _write(self, log: dict):
        print(json.dumps(log))
