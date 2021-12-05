from joatmon.database.model.log import Log
from joatmon.plugin.core import create
from joatmon.plugin.logger.core import Logger


class DatabaseLogger(Logger):
    def __init__(self, alias: str, level: str, database: str):
        super(DatabaseLogger, self).__init__(alias, level)

        self.database = database

    def _write(self, log: dict):
        with create(self.database) as db:
            db.save(Log(**log))
