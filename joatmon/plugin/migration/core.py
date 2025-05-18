import datetime
import uuid

from joatmon.core.utility import new_object_id
from joatmon.orm.document import (
    Document,
    create_new_type
)
from joatmon.orm.field import Field
from joatmon.orm.meta import Meta
from joatmon.plugin.core import Plugin


class Migration(Meta):
    __collection__ = '_migration'

    object_id = Field(uuid.UUID, nullable=False, default=new_object_id)
    table = Field(str, nullable=False)
    columns = Field(
        list, nullable=False, fields={
            'property': Field(str, nullable=False),
            'field': Field(
                dict, nullable=False, fields={
                    'name': Field(str, nullable=True),
                    'nullable': Field(bool, nullable=True),
                    'primary': Field(bool, nullable=True),
                }
            )
        }
    )
    datetime = Field(datetime.datetime, nullable=False)
    applied = Field(bool, nullable=False, default=False)


Migration = create_new_type(Migration, (Document,))


class MigrationPlugin(Plugin):
    async def create(self, document):
        raise NotImplementedError

    async def drop(self, document):
        raise NotImplementedError

    async def execute(self, document):
        raise NotImplementedError
