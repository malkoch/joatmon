from joatmon.core.utility import first_async

from joatmon import context
from joatmon.plugin.logger.core import LoggerPlugin


class DatabaseLogger(LoggerPlugin):
    def __init__(self, level: str, database: str, cls, language, ip):
        super(DatabaseLogger, self).__init__(level, language, ip)

        self.database = database
        self.cls = cls

    async def _write(self, log: dict):
        database = context.get_value(self.database)
        await first_async(database.insert(self.cls, log))
