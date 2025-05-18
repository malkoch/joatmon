import datetime

from joatmon.orm.document import (
    Document,
    create_new_type
)
from joatmon.orm.field import Field
from joatmon.orm.meta import Meta
from joatmon.plugin.core import Plugin


class Migration(Meta):
    __collection__ = '_migration'

    table = Field(str, nullable=False)
    columns = Field(str, nullable=False)
    datetime = Field(datetime.datetime, nullable=False)


Migration = create_new_type(Migration, (Document,))


class MigrationPlugin(Plugin):
    async def create(self, document):
        raise NotImplementedError

    async def drop(self, document):
        raise NotImplementedError

    async def execute(self, document):
        raise NotImplementedError
