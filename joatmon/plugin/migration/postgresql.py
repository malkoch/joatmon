import datetime
import json

from joatmon.core.utility import (
    JSONEncoder
)
from joatmon.plugin.migration.core import (
    Migration,
    MigrationPlugin
)


class PostgreSQLMigration(MigrationPlugin):
    def __init__(self, db):
        self.db = db

    async def create(self, document):
        collection = document.__metaclass__

        definition = Migration(
            **{
                'table': collection.__collection__,
                'columns': json.dumps(collection.fields(collection).values(), cls=JSONEncoder),
                'datetime': datetime.datetime.now()
            }
        )
        print(definition)


    async def drop(self, document):
        raise NotImplementedError

    async def execute(self, document):
        raise NotImplementedError
