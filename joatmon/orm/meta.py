import inspect
import typing
from datetime import datetime
from uuid import UUID

from joatmon.core.utility import (
    current_time,
    empty_object_id,
    get_converter,
    new_object_id
)
from joatmon.orm.constraint import (
    Constraint,
    UniqueConstraint
)
from joatmon.orm.field import Field
from joatmon.orm.index import Index


class Meta(type):
    __collection__ = 'meta'

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

    def __new__(mcs, name, bases, dct):
        return super().__new__(mcs, name, bases, dct)

    def fields(cls, predicate=lambda x: True) -> typing.Dict[str, Field]:
        return {k: v for k, v in inspect.getmembers(cls, lambda x: isinstance(x, Field)) if predicate(v)}

    def constraints(cls, predicate=lambda x: True) -> typing.Dict[str, Constraint]:
        return {k: v for k, v in inspect.getmembers(cls, lambda x: isinstance(x, Constraint)) if predicate(v)}

    def indexes(cls, predicate=lambda x: True) -> typing.Dict[str, Index]:
        return {k: v for k, v in inspect.getmembers(cls, lambda x: isinstance(x, Index)) if predicate(v)}

    def subclasses(cls, predicate=lambda x: True) -> typing.Dict[str, 'Meta']:
        subclasses = {}
        for subclass in cls.__subclasses__(cls):
            if subclass.__collection__ in subclasses:
                raise ValueError(f'{subclass.__collection__} is already in subclasses')
            subclasses[subclass.__collection__] = subclass

        return {k: v for k, v in subclasses.items() if predicate(v)}


def normalize_kwargs(meta, **kwargs):
    ret = {}

    fields = meta.fields(meta)
    for key in kwargs.keys():
        field = list(filter(lambda x: x[0] == key, fields.items()))
        if len(field) != 1:
            raise ValueError(f'field {key} has to be only one on the document')
        field = field[0][1]

        ret[key] = get_converter(field.dtype)(kwargs[key])
    return ret
