import functools
import inspect
import json
import os
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

email_pattern = re.compile(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)")
ip_address_pattern = re.compile(
    r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\."
    r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\."
    r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\."
    r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
)


def empty_object_id():
    return uuid.UUID('00000000-0000-0000-0000-000000000000')


def new_object_id():
    return uuid.uuid4()


def current_time():
    return datetime.now()


def new_nickname():
    return f'random_nickname_{uuid.uuid4()}'


def new_password():
    return f'random_password_{uuid.uuid4()}'


def mail_validator(email):
    return email_pattern.match(email) is not None


def ip_validator(ip):
    return ip_address_pattern.match(ip) is not None


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
    from joatmon.orm.enum import Enum

    if isinstance(value, dict):
        ret = {k: to_enumerable(v, string) for k, v in value.items()}
    elif isinstance(value, list):
        ret = [to_enumerable(v, string) for v in value]
    elif isinstance(value, tuple):
        ret = tuple([to_enumerable(v, string) for v in value])
    elif isinstance(value, Serializable):
        ret = to_enumerable({k: v for k, v in value.__dict__.items()}, string)
    elif isinstance(value, Enum):
        ret = str(value).lower()
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


def get_function_args(func, *args):
    if get_class_that_defined_method(func) is None:
        return args
    else:
        # need to check if the function is static / class method
        # if static no need to remove first args
        return args[1:]


def get_function_kwargs(func, **kwargs):
    return kwargs


def to_hash(func, *args, **kwargs):
    args_str = ", ".join([f"{arg}" for arg in get_function_args(func, *args)])
    kwargs_str = ", ".join([f"{k}={v}" for k, v in get_function_kwargs(func, **kwargs).items()])
    arg_kwarg_str = ', '.join(filter(lambda x: x != '', [args_str, kwargs_str]))
    return f'{func.__module__}.{func.__qualname__}({arg_kwarg_str})'


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
        set: _set_converter,
        object: _object_converter
    }

    return converters.get(kind, _object_converter)


def to_list(items):
    ret = []
    for item in iter(items):
        ret.append(item)
    return ret


async def to_list_async(items):
    ret = []
    async for item in aiter(items):
        ret.append(item)
    return ret


def first(items):
    try:
        iterable = iter(items)
        first_item = next(iterable)
        return first_item
    except StopIteration:
        return None


async def first_async(items):
    try:
        iterable = aiter(items)
        first_item = await anext(iterable)
        return first_item
    except StopAsyncIteration:
        return None


def single(items):
    try:
        iterable = iter(items)
        first_item = next(iterable)
    except StopIteration:
        return None

    try:
        next(iterable)
        return None
    except StopIteration:
        return first_item


async def single_async(items):
    try:
        iterable = aiter(items)
        first_item = await anext(iterable)
    except StopAsyncIteration:
        return None

    try:
        await anext(iterable)
        return None
    except StopAsyncIteration:
        return first_item


def pretty_printer(headers, m=None):
    def get_max_size(idx):
        max_size = m or os.get_terminal_size().columns
        weight = headers[idx][1]
        # if idx == len(headers) - 1:
        #     return max_size - (int(max_size * (weight / sum(x[1] for x in headers))) * len(headers) - 1)
        # else:
        #     return int(max_size * (weight / sum(x[1] for x in headers)))
        return int(max_size * (weight / sum(x[1] for x in headers)))

    def left_padding(idx, value):
        max_size = get_max_size(idx)
        return (max_size - len(value)) // 2

    def right_padding(idx, value):
        max_size = get_max_size(idx)
        return max_size - (((max_size - len(value)) // 2) + len(value))

    def prettify_header(idx, value):
        if len(value) > get_max_size(idx):
            return value[:get_max_size(idx)]

        return ' ' * left_padding(idx, value) + value + ' ' * right_padding(idx, value)

    def prettify_value(idx, value):
        if len(value) > get_max_size(idx):
            return value[:get_max_size(idx)]

        return ' ' * left_padding(idx, value) + value + ' ' * right_padding(idx, value)

    def pretty_print(values):
        return ' '.join([prettify_value(idx, value) for idx, value in enumerate(values)])

    return ' '.join([prettify_header(idx, header) for idx, (header, max_size) in enumerate(headers)]), pretty_print


def convert_size(size_bytes):
    import math
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


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


def get_module_classes(module):
    return inspect.getmembers(module, inspect.isclass)


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
