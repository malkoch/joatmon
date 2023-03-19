import inspect
import typing

from joatmon.core.utility import get_converter
from joatmon.orm.constraint import Constraint
from joatmon.orm.field import Field
from joatmon.orm.index import Index


class Meta(type):
    __collection__ = 'meta'

    structured = True
    encrypt = False

    qb = None

    def __new__(mcs, name, bases, dct):
        return super().__new__(mcs, name, bases, dct)

    def fields(cls, predicate=lambda x: True) -> typing.Dict[str, Field]:
        return {k: v for k, v in inspect.getmembers(cls, lambda x: isinstance(x, Field)) if predicate(v)}

    def constraints(cls, predicate=lambda x: True) -> typing.Dict[str, Constraint]:
        return {k: v for k, v in inspect.getmembers(cls, lambda x: isinstance(x, Constraint)) if predicate(v)}

    def indexes(cls, predicate=lambda x: True) -> typing.Dict[str, Index]:
        return {k: v for k, v in inspect.getmembers(cls, lambda x: isinstance(x, Index)) if predicate(v)}

    def query(cls):
        return cls.qb


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
