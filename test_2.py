from datetime import datetime
from uuid import UUID

from joatmon.orm.constraint import UniqueConstraint
from joatmon.orm.document import (
    create_new_type,
    Document
)
from joatmon.orm.field import Field
from joatmon.orm.meta import Meta
from joatmon.utility import (
    current_time,
    empty_object_id,
    new_object_id
)


class Structured(Meta):
    structured = True

    object_id = Field(UUID, nullable=False, default=new_object_id, primary=True)
    created_at = Field(datetime, nullable=False, default=current_time)
    creator_id = Field(UUID, nullable=False, default=empty_object_id)
    updated_at = Field(datetime, nullable=False, default=current_time)
    updater_id = Field(UUID, nullable=False, default=empty_object_id)
    deleted_at = Field(datetime, nullable=True, default=current_time)
    deleter_id = Field(UUID, nullable=True, default=empty_object_id)
    is_deleted = Field(bool, nullable=False, default=False)

    unique_constraint_object_id = UniqueConstraint('object_id')


class Type(Structured):
    __collection__ = 'type'

    code = Field(int, nullable=False)
    name = Field(str, nullable=False, resource=True)
    description = Field(str, nullable=False, resource=True)


Type = create_new_type(Type, (Document,))

t = Type()
t.code = 12
t.name = 'test_n'
t.description = 'test_d'

print(t)
