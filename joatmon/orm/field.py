import typing
import uuid
from datetime import datetime
from time import (
    mktime,
    struct_time
)

from joatmon.serializable import Serializable
from joatmon.utility import to_enumerable


class Field(Serializable):
    def __init__(
            self,
            dtype: typing.Union[type, typing.List, typing.Tuple],
            nullable: bool = True,
            default=None,
            primary: bool = False,
            encrypt: bool = False,
            hash_: bool = False,
            resource: str = None,
            fields: dict = None
    ):
        super(Field, self).__init__()

        self.dtype = dtype
        self.nullable = nullable
        self.primary = primary
        self.encrypt = encrypt
        self.hash_ = hash_

        if not callable(default):
            self.default = lambda: default
        else:
            self.default = default

        self.encrypted = False
        self.hashed = False
        self.resource = resource
        self.fields = fields or {}


def get_converter(field: Field):
    def _datetime_converter(value: object) -> typing.Optional[datetime]:
        if value is None:
            return value

        if isinstance(value, datetime):
            return value

        if isinstance(value, struct_time):
            return datetime.fromtimestamp(mktime(value))

        if isinstance(value, str):
            return datetime.fromisoformat(value)

        raise ValueError(f'cannot convert {value} with type {type(value)} to datetime')

    def _integer_converter(value: object) -> typing.Optional[int]:
        if value is None:
            return value

        if isinstance(value, float):
            return int(value)

        if isinstance(value, int):
            return int(value)

        if isinstance(value, str):
            return int(value)

        raise ValueError(f'cannot convert {value} with type {type(value)} to integer')

    def _float_converter(value: object) -> typing.Optional[float]:
        if value is None:
            return value

        if isinstance(value, float):
            return value

        if isinstance(value, int):
            return float(value)

        if isinstance(value, str):
            return float(value)

        raise ValueError(f'cannot convert {value} with type {type(value)} to float')

    def _string_converter(value: object) -> typing.Optional[str]:
        if value is None:
            return value

        if isinstance(value, str):
            if field.resource is not None:
                value = field.resource.format(value)
            return value

        # need to convert bytes as well

        raise ValueError(f'cannot convert {value} with type {type(value)} to string')

    def _byte_converter(value: object) -> typing.Optional[bytes]:
        if value is None:
            return value

        if isinstance(value, bytes):
            return value

        # need to convert string and int list as well

        raise ValueError(f'cannot convert {value} with type {type(value)} to bytes')

    def _boolean_converter(value: object) -> typing.Optional[bool]:
        if value is None:
            return value

        if isinstance(value, bool):
            return value

        # need to convert float, int and str as well

        raise ValueError(f'cannot convert {value} with type {type(value)} to boolean')

    def _uuid_converter(value: object) -> typing.Optional[uuid.UUID]:
        if value is None:
            return value

        if isinstance(value, uuid.UUID):
            return value

        if isinstance(value, str):
            return uuid.UUID(value)

        raise ValueError(f'cannot convert {value} with type {type(value)} to uuid')

    def _dictionary_converter(value: object) -> typing.Optional[dict]:
        if value is None:
            return value

        if isinstance(value, dict):
            return to_enumerable(value)

        if hasattr(value, 'dict'):
            return to_enumerable(value)

        raise ValueError(f'cannot convert {value} with type {type(value)} to dictionary')

    def _list_converter(value: object) -> typing.Optional[list]:
        if value is None:
            return value

        # if isinstance(value, str):
        #     value = json.loads(value)

        if isinstance(value, (list, tuple, set)):
            ret = []
            for v in value:
                if field.fields is not None:
                    obj = {}
                    for field_name, field_meta in field.fields.items():
                        obj[field_name] = get_converter(field_meta)(v[field_name])
                    ret.append(obj)
                else:
                    ret.append(to_enumerable(v))
            return ret

        raise ValueError(f'cannot convert {value} with type {type(value)} to list')

    def _tuple_converter(value: object) -> typing.Optional[tuple]:
        if value is None:
            return value

        # if isinstance(value, str):
        #     value = json.loads(value)

        if isinstance(value, (list, tuple, set)):
            ret = []
            for v in value:
                ret.append(to_enumerable(v))
            ret = tuple(ret)
            return ret

        raise ValueError(f'cannot convert {value} with type {type(value)} to tuple')

    def _set_converter(value: object) -> typing.Optional[set]:
        if value is None:
            return value

        # if isinstance(value, str):
        #     value = json.loads(value)

        if isinstance(value, (list, tuple, set)):
            ret = []
            for v in value:
                ret.append(to_enumerable(v))
            ret = set(ret)
            return ret

        raise ValueError(f'cannot convert {value} with type {type(value)} to tuple')

    def _object_converter(value: object) -> typing.Optional[object]:
        if value is None:
            return value

        if isinstance(value, dict):
            if field.fields is not None:
                obj = {}
                for field_name, field_meta in field.fields.items():
                    obj[field_name] = get_converter(field_meta)(value[field_name])
                return _dictionary_converter(obj)
            else:
                return _dictionary_converter(value)

            # return _dictionary_converter(value)

        if isinstance(value, list):
            return _list_converter(value)

        if isinstance(value, tuple):
            return _tuple_converter(value)

        if isinstance(value, str):
            return _string_converter(value)

        if hasattr(value, 'dict'):
            return value.dict

        return value

    converters = {
        datetime: _datetime_converter,
        int: _integer_converter,
        float: _float_converter,
        str: _string_converter,
        bytes: _byte_converter,
        bool: _boolean_converter,
        uuid.UUID: _uuid_converter,
        dict: _dictionary_converter,
        list: _list_converter,
        tuple: _tuple_converter,
        set: _set_converter,
        object: _object_converter
    }

    return converters.get(field.dtype, _object_converter)
