import inspect
import typing
from datetime import datetime
from uuid import UUID

from joatmon.orm.constraint import (
    Constraint,
    UniqueConstraint
)
from joatmon.orm.field import Field
from joatmon.orm.index import Index
from joatmon.orm.utility import (
    current_time,
    empty_object_id,
    new_object_id
)


class Meta(type):
    __collection__ = 'meta'

    object_id = Field(UUID, nullable=False, fallback=new_object_id)
    created_at = Field(datetime, nullable=False, fallback=current_time)
    creator_id = Field(UUID, nullable=False, fallback=empty_object_id)
    updated_at = Field(datetime, nullable=False, fallback=current_time)
    updater_id = Field(UUID, nullable=False, fallback=empty_object_id)
    deleted_at = Field(datetime, nullable=True, fallback=current_time)
    deleter_id = Field(UUID, nullable=True, fallback=empty_object_id)
    is_deleted = Field(bool, nullable=False, fallback=False)

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
