import typing

from joatmon.core import CoreException
from joatmon.database.meta import Meta
from joatmon.serializable import Serializable
from joatmon.utility import get_converter


class DocumentException(CoreException):
    pass


class Document(Serializable):  # need to have copy and deepcopy functions as well
    __metaclass__ = Meta

    def __init__(self, **kwargs):
        super(Document, self).__init__(**kwargs)

        for name, field in self.__metaclass__.fields(self.__metaclass__).items():
            if name not in kwargs:
                setattr(self, name, None)

    def __getattr__(self, item):
        return self.__dict__.get(item, None)

    def __setattr__(self, key, value):
        if key not in self.__metaclass__.fields(self.__metaclass__).keys():
            self.__dict__[key] = value
            return

        field = self.__metaclass__.fields(self.__metaclass__)[key]
        self.__dict__[key] = get_converter(field.kind)(value)

    def validate(self):
        ret = {}
        for name, field in self.__metaclass__.fields(self.__metaclass__).items():
            value = getattr(self, name, None)

            default_value = None
            if field.fallback is not None and callable(field.fallback):
                default_value = field.fallback()
            elif field.fallback is not None and not callable(field.fallback):
                default_value = field.fallback

            if value is None and not field.nullable:
                setattr(self, name, default_value)

            value = getattr(self, name, None)

            if value is None and not field.nullable:
                raise ValueError(f'field {name} is not nullable')

            if isinstance(field.kind, (tuple, list)):
                if ((value is not None and field.nullable) or not field.nullable) and not isinstance(value, field.kind):
                    raise ValueError(f'field {name} has to be one of the following {field.kind} not {type(value).__name__}')
            else:
                if ((value is not None and field.nullable) or not field.nullable) and type(value) is not field.kind:
                    raise ValueError(f'field {name} has to be type {field.kind} not {type(value).__name__}')

            constraints = self.__metaclass__.constraints(self.__metaclass__).values()
            constraints = list(filter(lambda x: x.field == name, constraints))

            if not field.nullable and constraints is not None:
                for constraint in constraints:
                    constraint.check(value)

            ret[name] = getattr(self, name, None)

        return ret
