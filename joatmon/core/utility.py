import base64
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

email_pattern = re.compile(r'(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)')
ip_address_pattern = re.compile(
    r'(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.'
    r'(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.'
    r'(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.'
    r'(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
)


def empty_object_id():
    """
    Convert the Serializable object to a dictionary with Pascal case keys.

    Returns:
        dict: The dictionary representation of the Serializable object with Pascal case keys.
    """
    return uuid.UUID('00000000-0000-0000-0000-000000000000')


def new_object_id():
    """
    Generate a new UUID.

    Returns:
        UUID: A new UUID.
    """
    return uuid.uuid4()


def current_time():
    """
    Get the current datetime.

    Returns:
        datetime: The current datetime.
    """
    return datetime.now()


def new_nickname():
    """
    Generate a new random nickname.

    Returns:
        str: A new random nickname.
    """
    return f'random_nickname_{uuid.uuid4()}'


def new_password():
    """
    Generate a new random password.

    Returns:
        str: A new random password.
    """
    return f'random_password_{uuid.uuid4()}'


def mail_validator(email):
    """
    Validate an email address.

    Args:
        email (str): The email address to validate.

    Returns:
        bool: True if the email address is valid, False otherwise.
    """
    return email_pattern.match(email) is not None


def ip_validator(ip):
    """
    Validate an IP address.

    Args:
        ip (str): The IP address to validate.

    Returns:
        bool: True if the IP address is valid, False otherwise.
    """
    return ip_address_pattern.match(ip) is not None


def to_snake_string(string: str):
    """
    Convert a string to snake case.

    Args:
        string (str): The string to convert.

    Returns:
        str: The string in snake case.
    """
    return pascal_case_pattern.sub('_', string).lower()


def to_pascal_string(string: str):
    """
    Convert a string to Pascal case.

    Args:
        string (str): The string to convert.

    Returns:
        str: The string in Pascal case.
    """
    return ''.join(word.title() for word in string.split('_'))


def to_upper_string(string: str):
    """
    Convert a string to upper case.

    Args:
        string (str): The string to convert.

    Returns:
        str: The string in upper case.
    """
    return string.upper()


def to_lower_string(string: str):
    """
    Convert a string to lower case.

    Args:
        string (str): The string to convert.

    Returns:
        str: The string in lower case.
    """
    return string.lower()


def to_title(string: str):
    """
    Convert a string to title case.

    Args:
        string (str): The string to convert.

    Returns:
        str: The string in title case.
    """
    return string.title()


def to_enumerable(value, string=False):
    """
    Convert a value to an enumerable.

    Args:
        value (Any): The value to convert.
        string (bool): If True, convert the value to a string.

    Returns:
        Any: The value as an enumerable.
    """
    if value is None:
        return None

    from joatmon.core.serializable import Serializable
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
    """
    Convert a value to a specific case.

    Args:
        case (str): The case to convert to.
        value (Any): The value to convert.
        key (str, optional): The key to convert. Defaults to None.
        convert_value (bool, optional): If True, convert the value to a string. Defaults to False.

    Returns:
        Any: The value converted to the specified case.
    """
    from joatmon.core.serializable import Serializable

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
    """
    Get the arguments of a function.

    Args:
        func (function): The function to get the arguments of.
        *args: The arguments of the function.

    Returns:
        tuple: The arguments of the function.
    """
    if get_class_that_defined_method(func) is None:
        return args
    else:
        # need to check if the function is static / class method
        # if static no need to remove first args
        return args[1:]


def get_function_kwargs(func, **kwargs):
    """
    Get the keyword arguments of a function.

    Args:
        func (function): The function to get the keyword arguments of.
        **kwargs: The keyword arguments of the function.

    Returns:
        dict: The keyword arguments of the function.
    """
    return kwargs


def to_hash(func, *args, **kwargs):
    """
    Generate a hash for a function and its arguments.

    Args:
        func (function): The function to generate a hash for.
        *args: The arguments of the function.
        **kwargs: The keyword arguments of the function.

    Returns:
        str: The hash of the function and its arguments.
    """
    args_str = ', '.join([f'{arg}' for arg in get_function_args(func, *args)])
    kwargs_str = ', '.join([f'{k}={v}' for k, v in get_function_kwargs(func, **kwargs).items()])
    arg_kwarg_str = ', '.join(filter(lambda x: x != '', [args_str, kwargs_str]))
    return f'{func.__module__}.{func.__qualname__}({arg_kwarg_str})'


def get_converter(kind: type):
    """
    Get a converter for a specific type.

    Args:
        kind (type): The type to get a converter for.

    Returns:
        function: The converter for the specified type.
    """

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
        object: _object_converter,
    }

    return converters.get(kind, _object_converter)


def to_list(items):
    """
    Convert an iterable to a list.

    Args:
        items (iterable): The iterable to convert.

    Returns:
        list: The list representation of the iterable.
    """
    ret = []
    for item in iter(items):
        ret.append(item)
    return ret


async def to_list_async(items):
    """
    Asynchronously convert an iterable to a list.

    Args:
        items (iterable): The iterable to convert.

    Returns:
        list: The list representation of the iterable.
    """
    ret = []
    async for item in aiter(items):
        ret.append(item)
    return ret


def first(items):
    """
    Get the first item of an iterable.

    Args:
        items (iterable): The iterable to get the first item from.

    Returns:
        Any: The first item of the iterable, or None if the iterable is empty.
    """
    try:
        iterable = iter(items)
        first_item = next(iterable)
        return first_item
    except StopIteration:
        return None


async def first_async(items):
    """
    Asynchronously get the first item of an iterable.

    Args:
        items (iterable): The iterable to get the first item from.

    Returns:
        Any: The first item of the iterable, or None if the iterable is empty.
    """
    try:
        iterable = aiter(items)
        first_item = await anext(iterable)
        return first_item
    except StopAsyncIteration:
        return None


def single(items):
    """
    Get the single item of an iterable.

    Args:
        items (iterable): The iterable to get the single item from.

    Returns:
        Any: The single item of the iterable, or None if the iterable is empty or contains more than one item.
    """
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
    """
    Asynchronously get the single item of an iterable.

    Args:
        items (iterable): The iterable to get the single item from.

    Returns:
        Any: The single item of the iterable, or None if the iterable is empty or contains more than one item.
    """
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
    """
    Create a pretty printer for a list of headers.

    Args:
        headers (list): The headers to pretty print.
        m (int, optional): The maximum size of the pretty printer. Defaults to the terminal size.

    Returns:
        function: A function that pretty prints values according to the headers.
    """

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
            return value[: get_max_size(idx)]

        return ' ' * left_padding(idx, value) + value + ' ' * right_padding(idx, value)

    def prettify_value(idx, value):
        if len(value) > get_max_size(idx):
            return value[: get_max_size(idx)]

        return ' ' * left_padding(idx, value) + value + ' ' * right_padding(idx, value)

    def pretty_print(values):
        return ' '.join([prettify_value(idx, value) for idx, value in enumerate(values)])

    return ' '.join([prettify_header(idx, header) for idx, (header, max_size) in enumerate(headers)]), pretty_print


def convert_size(size_bytes):
    """
    Convert a size in bytes to a human-readable string.

    Args:
        size_bytes (int): The size in bytes.

    Returns:
        str: The human-readable string representation of the size.
    """
    import math

    if size_bytes == 0:
        return '0 B'
    size_name = ('B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB')
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return '%s %s' % (s, size_name[i])


def get_class_that_defined_method(meth):
    """
    Get the class that defined a method.

    Args:
        meth (function): The method to get the defining class of.

    Returns:
        type: The class that defined the method, or None if the method is not a method of a class.
    """
    if isinstance(meth, functools.partial):
        return get_class_that_defined_method(meth.func)
    if inspect.ismethod(meth) or (
            inspect.isbuiltin(meth)
            and getattr(meth, '__self__', None) is not None
            and getattr(meth.__self__, '__class__', None)
    ):
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
    """
    Get the functions of a module.

    Args:
        module (module): The module to get the functions of.

    Returns:
        list: A list of tuples, where each tuple contains the name of a function and the function itself.
    """
    return inspect.getmembers(module, inspect.isfunction)


def get_module_classes(module):
    """
    Get the classes of a module.

    Args:
        module (module): The module to get the classes of.

    Returns:
        list: A list of tuples, where each tuple contains the name of a class and the class itself.
    """
    return inspect.getmembers(module, inspect.isclass)


class JSONEncoder(json.JSONEncoder):
    """
    A JSON encoder that can handle more Python data types than the standard json.JSONEncoder.

    This encoder can handle datetime, date, time, timedelta, uuid.UUID, bytes, and callable objects in addition to the types that json.JSONEncoder can handle.
    """

    def default(self, obj):
        """
        Implement this method in a subclass such that it returns a serializable object for `obj`, or calls the base implementation (to raise a TypeError).

        Args:
            obj (Any): The object to convert to a serializable object.

        Returns:
            Any: A serializable object.
        """
        if isinstance(obj, (datetime, date, time)):
            return obj.isoformat()
        if isinstance(obj, timedelta):
            return (datetime.min + obj).time().isoformat()
        if isinstance(obj, uuid.UUID):
            return str(obj)
        if isinstance(obj, bytes):
            return base64.b64encode(obj).decode('utf-8')
        if callable(obj):
            return str(obj)
        return str(obj)


class JSONDecoder(json.JSONDecoder):
    """
    A JSON decoder that uses a custom object hook.

    This decoder uses an object hook that simply returns the input dictionary without any modifications.
    """

    def __init__(self, *args, **kwargs):
        super(JSONDecoder, self).__init__(object_hook=self.object_hook, *args, **kwargs)

    @staticmethod
    def object_hook(d):
        """
        Implement this method in a subclass such that it returns a Python object for `d`.

        Args:
            d (dict): The dictionary to convert to a Python object.

        Returns:
            Any: A Python object.
        """
        return d
