import datetime

from joatmon.core import context
from joatmon.core.serializable import Serializable
from joatmon.core.utility import (
    to_enumerable,
    to_list_async
)
from joatmon.plugin.migration.core import (
    Migration,
    MigrationPlugin
)


class PostgreSQLMigration(MigrationPlugin):
    def __init__(self, history, database):
        self.history = history
        self.database = database

    async def create(self, document):
        collection = document.__metaclass__

        definition = Migration(
            **{
                'table': collection.__collection__,
                'columns': [{'property': name, 'field': to_enumerable(field)} for name, field in collection.fields(collection).items()],
                'datetime': datetime.datetime.now(),
                'applied': False
            }
        )

        print(Serializable.from_json(definition.json))

        db = context.get_value(self.history)
        await db.insert(Migration, Serializable.from_json(definition.json))


    async def drop(self, document):
        db = context.get_value(self.history)
        history = await to_list_async(db.read(Migration, {'table': document.__metaclass__.__collection__, 'applied': False}))

        for migration in history:
            await db.delete(Migration, migration.object_id)

    async def execute(self, document):
        db = context.get_value(self.history)
        history = await to_list_async(db.read(Migration, {'table': document.__metaclass__.__collection__, 'applied': False}))

        for migration in history:
            await db.update(Migration, {'object_id': migration.object_id}, {'applied': True})
