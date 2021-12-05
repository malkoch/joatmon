import functools
import inspect
import json
import re
import typing
import uuid
from datetime import (
    date,
    datetime,
    time,
    timedelta
)
from time import (
    mktime,
    struct_time
)
from uuid import UUID

pascal_case_pattern = re.compile(r'(?<!^)(?=[A-Z])')


def to_snake_string(string: str):
    return pascal_case_pattern.sub('_', string).lower()


def to_pascal_string(string: str):
    return ''.join(word.title() for word in string.split('_'))


def to_upper_string(string: str):
    return string.upper()


def to_lower_string(string: str):
    return string.lower()


def to_title(string: str):
    return string.title()


def to_enumerable(value, string=False):
    if value is None:
        return None

    from joatmon.serializable import Serializable
    if isinstance(value, dict):
        ret = {k: to_enumerable(v, string) for k, v in value.items()}
    elif isinstance(value, list):
        ret = [to_enumerable(v, string) for v in value]
    elif isinstance(value, tuple):
        ret = tuple([to_enumerable(v, string) for v in value])
    elif isinstance(value, Serializable):
        ret = to_enumerable({k: v for k, v in value.__dict__.items()}, string)
    else:
        ret = value
        if string:
            ret = str(ret)
    return ret


def to_case(case, value, key=None, convert_value=False):
    from joatmon.serializable import Serializable

    enumerable = to_enumerable(value)

    if case == 'snake':
        new_key = to_snake_string(key) if key is not None else None
    elif case == 'pascal':
        new_key = to_pascal_string(key) if key is not None else None
    elif case == 'upper':
        new_key = to_upper_string(key) if key is not None else None
    elif case == 'lower':
        new_key = to_lower_string(key) if key is not None else None
    else:
        raise ValueError(f'{case} is not supported case.')

    if isinstance(enumerable, dict):
        ret = new_key, {k: v for k, v in [to_case(case, v, k) for k, v in enumerable.items()]}
    elif isinstance(enumerable, list):
        ret = new_key, [to_case(case, v) for v in enumerable]
    elif isinstance(enumerable, tuple):
        ret = new_key, tuple([to_case(case, v) for v in enumerable])
    elif isinstance(value, Serializable):
        ret = new_key, to_case(case, value.__dict__)
    else:
        if convert_value:
            value = to_snake_string(value)
        ret = new_key, value

    if new_key is None:
        return ret[1]
    return ret


def to_hash(func, *args, **kwargs):
    return f'{func.__module__}.{func.__name__}({", ".join(f"{arg}" for arg in args)}, {", ".join([f"{k}={v}" for k, v in kwargs.items()])})'


def get_converter(kind: type):
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
            return value

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

        if isinstance(value, list):
            ret = []
            for v in value:
                ret.append(to_enumerable(v))
            return ret

        raise ValueError(f'cannot convert {value} with type {type(value)} to list')

    def _tuple_converter(value: object) -> typing.Optional[tuple]:
        if value is None:
            return value

        if isinstance(value, tuple):
            ret = []
            for v in value:
                ret.append(to_enumerable(v))
            ret = tuple(ret)
            return ret

        raise ValueError(f'cannot convert {value} with type {type(value)} to tuple')

    def _object_converter(value: object) -> typing.Optional[object]:
        if value is None:
            return value

        if isinstance(value, dict):
            return _dictionary_converter(value)

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
        UUID: _uuid_converter,
        dict: _dictionary_converter,
        list: _list_converter,
        tuple: _tuple_converter,
        object: _object_converter,
    }

    return converters.get(kind, _object_converter)


def first(items):
    try:
        iterable = iter(items)
        first_item = next(iterable)
        return first_item
    except TypeError:
        return None
    except StopIteration:
        return None


def single(items):
    try:
        iterable = iter(items)
        first_item = next(iterable)
    except TypeError:
        return None
    except StopIteration:
        return None

    try:
        next(iterable)
        return None
    except TypeError:
        return first_item
    except StopIteration:
        return first_item


def get_class_that_defined_method(meth):
    if isinstance(meth, functools.partial):
        return get_class_that_defined_method(meth.func)
    if inspect.ismethod(meth) or (inspect.isbuiltin(meth) and getattr(meth, '__self__', None) is not None and getattr(meth.__self__, '__class__', None)):
        for cls in inspect.getmro(meth.__self__.__class__):
            if meth.__name__ in cls.__dict__:
                return cls
        meth = getattr(meth, '__func__', meth)  # fallback to __qualname__ parsing
    if inspect.isfunction(meth):
        cls = getattr(inspect.getmodule(meth), meth.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0], None)
        if isinstance(cls, type):
            return cls
    return getattr(meth, '__objclass__', None)  # handle special descriptor objects


def get_module_functions(module):
    return inspect.getmembers(module, inspect.isfunction)


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date, time)):
            return obj.isoformat()
        if isinstance(obj, timedelta):
            return (datetime.min + obj).time().isoformat()
        if isinstance(obj, uuid.UUID):
            return str(obj)
        if callable(obj):
            return str(obj)
        return str(obj)


class JSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super(JSONDecoder, self).__init__(object_hook=self.object_hook, *args, **kwargs)

    @staticmethod
    def object_hook(d):
        return d
        # ret = {}
        # for key, value in d.items():
        #     print('__type__' in value, key)
        #     if '__type__' not in value:
        #         ret[key] = value
        #         continue
        #
        #     t = value.pop('__type__')
        #     try:
        #         if t == 'datetime':
        #             ret[key] = datetime(**value)
        #     except:
        #         value['__type___'] = t
        #         ret[key] = value
        # return ret
