import json
import pickle

from joatmon.utility import (
    JSONEncoder,
    to_case,
    to_enumerable
)


class Serializable(object):
    def __init__(self, **kwargs):
        for slot_name, slot_value in kwargs.items():
            if not isinstance(slot_name, str):
                raise ValueError(f'{slot_name} is type {type(slot_name)} is not supported. only string type is supported for field names.')
            setattr(self, slot_name, slot_value)

    def __str__(self):
        return self.pretty_json

    def __repr__(self):
        return str(self)

    def keys(self):
        for key in self.__dict__.keys():
            yield key

    def values(self):
        for value in self.__dict__.values():
            yield value

    def items(self):
        for key, value in self.__dict__.items():
            yield key, value

    def __getitem__(self, key):
        return self.__dict__[key]

    @property
    def dict(self):
        return to_enumerable(self)

    @classmethod
    def from_dict(cls, dictionary: dict):
        return cls(**dictionary)

    @property
    def json(self) -> str:
        return json.dumps(self.dict, cls=JSONEncoder)

    @property
    def pretty_json(self) -> str:
        return json.dumps(self.dict, cls=JSONEncoder, indent=4)

    @classmethod
    def from_json(cls, string: str):
        return cls.from_dict(json.loads(string))

    @property
    def bytes(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def from_bytes(cls, value: bytes):
        if value is None:
            return None

        obj = pickle.loads(value)
        return obj

    @property
    def snake(self):
        return to_case('snake', self.__dict__)

    @property
    def pascal(self):
        return to_case('pascal', self.__dict__)
