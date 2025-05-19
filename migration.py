import asyncio
from datetime import datetime
from uuid import UUID

from joatmon.core import context
from joatmon.core.utility import (
    current_time,
    empty_object_id,
    new_object_id
)
from joatmon.orm.constraint import UniqueConstraint
from joatmon.orm.document import (
    Document,
    create_new_type
)
from joatmon.orm.field import Field
from joatmon.orm.meta import Meta
from joatmon.plugin.core import register
from joatmon.plugin.database.postgresql import PostgreSQLDatabase


register(PostgreSQLDatabase, 'database', '10.2.7.202', 5432, 'postgres', '', 'appyfany')


def get_current_user_id():
    return empty_object_id()


class Structured(Meta):
    structured = True
    force = True

    object_id = Field(UUID, nullable=False, default=new_object_id, primary=True)
    created_at = Field(datetime, nullable=False, default=current_time)
    creator_id = Field(UUID, nullable=False, default=get_current_user_id)
    updated_at = Field(datetime, nullable=False, default=current_time)
    updater_id = Field(UUID, nullable=False, default=get_current_user_id)
    deleted_at = Field(datetime, nullable=True)
    deleter_id = Field(UUID, nullable=True)
    is_deleted = Field(bool, nullable=False, default=False)

    unique_constraint_object_id = UniqueConstraint('object_id')


class DAttributeType(Structured):
    __collection__ = 'd_attribute_type'

    code = Field(int, nullable=False)
    name = Field(str, nullable=False, resource=True)
    description = Field(str, nullable=False, resource=True)


DAttributeType = create_new_type(DAttributeType, (Document,))


async def initialize():
    database = context.get_value('database')
    await database.create(DAttributeType)
    await database.alter(DAttributeType)
    await database.insert(DAttributeType, {'code': 1, 'name': 'Basic', 'description': 'Basic'})


asyncio.run(initialize())
