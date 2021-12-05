from datetime import datetime

from joatmon.plugin.logger.core import Logger


class TerminalLogger(Logger):
    def __init__(self, alias: str, level: str):
        super(TerminalLogger, self).__init__(alias, level)

    def _write(self, log: dict):
        message = f'Level: {log["level"]} - Timestamp: {datetime.utcnow().isoformat()} - Message: {log}'
        print(message)
