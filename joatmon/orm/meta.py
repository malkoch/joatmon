import inspect
import typing
from collections import OrderedDict

from joatmon.core.utility import get_converter
from joatmon.orm.constraint import Constraint
from joatmon.orm.field import Field
from joatmon.orm.index import Index
from joatmon.orm.key import Key


class Meta(type):
    """
    Metaclass for ORM system. It provides methods to access fields, constraints, and indexes of a class.

    Attributes:
        __collection__ (str): The collection name in the database.
        structured (bool): Whether the class is structured.
        force (bool): Whether to force the structure.
        qb: The query builder for the class.
    """

    __collection__ = 'meta'

    structured = True
    force = True

    qb = None

    def __new__(mcs, name, bases, dct):
        """
        Creates a new instance of the Meta class.

        Args:
            name (str): The name of the class.
            bases (tuple): The base classes of the class.
            dct (dict): The dictionary of class attributes.

        Returns:
            Meta: A new instance of the Meta class.
        """
        return super().__new__(mcs, name, bases, dct)

    def validate(cls):
        # make sure there is only one fields with primary key tagged
        field_names = set()

        for field_name, field in cls.fields(cls).items():
            if field_name in field_names:
                raise ValueError(f'field {field_name} is duplicated')
            field_names.add(field_name)

            for name in field.names:
                prev, new = name.split('->')
                if new in field_names:
                    raise ValueError(f'field {new} is duplicated')
                field_names.add(new)

    def fields(cls, predicate=lambda x: True) -> typing.Dict[str, Field]:
        """
        Gets the fields of the class.

        Args:
            predicate (callable): A function that determines which fields to include.

        Returns:
            dict: A dictionary of the fields of the class.
        """
        ret = OrderedDict()
        for _cls in reversed(cls.mro(cls)):
            for k, v in vars(_cls).items():
                if not isinstance(v, Field) or not predicate(v):
                    continue
                ret[k] = v
        return ret

    def constraints(cls, predicate=lambda x: True) -> typing.Dict[str, Constraint]:
        """
        Gets the constraints of the class.

        Args:
            predicate (callable): A function that determines which constraints to include.

        Returns:
            dict: A dictionary of the constraints of the class.
        """
        return {k: v for k, v in inspect.getmembers(cls, lambda x: isinstance(x, Constraint)) if predicate(v)}

    def indexes(cls, predicate=lambda x: True) -> typing.Dict[str, Index]:
        """
        Gets the indexes of the class.

        Args:
            predicate (callable): A function that determines which indexes to include.

        Returns:
            dict: A dictionary of the indexes of the class.
        """
        return {k: v for k, v in inspect.getmembers(cls, lambda x: isinstance(x, Index)) if predicate(v)}

    def keys(cls, predicate=lambda x: True) -> typing.Dict[str, Index]:
        """
        Gets the indexes of the class.

        Args:
            predicate (callable): A function that determines which indexes to include.

        Returns:
            dict: A dictionary of the indexes of the class.
        """
        return {k: v for k, v in inspect.getmembers(cls, lambda x: isinstance(x, Key)) if predicate(v)}

    def query(cls):
        """
        Gets the query builder for the class.

        Returns:
            The query builder for the class.
        """
        return cls.qb


def normalize_kwargs(meta, **kwargs):
    """
    Normalizes the keyword arguments to match the fields of the class.

    Args:
        meta (Meta): The metaclass of the class.
        **kwargs: The keyword arguments to normalize.

    Returns:
        dict: A dictionary of the normalized keyword arguments.
    """
    ret = {}

    fields = meta.fields(meta)
    for key in kwargs.keys():
        field = list(filter(lambda x: x[0] == key, fields.items()))
        if len(field) != 1:
            raise ValueError(f'field {key} has to be only one on the document')
        field = field[0][1]

        ret[key] = get_converter(field.dtype)(kwargs[key])
    return ret
