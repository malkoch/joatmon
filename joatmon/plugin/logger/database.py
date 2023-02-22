from joatmon import context
from joatmon.plugin.logger.core import LoggerPlugin


class DatabaseLogger(LoggerPlugin):
    def __init__(self, level: str, database: str, cls):
        super(DatabaseLogger, self).__init__(level)

        self.database = database
        self.cls = cls

    async def _write(self, log: dict):
        database = context.get_value(self.database)
        await database.insert(self.cls(**log))
