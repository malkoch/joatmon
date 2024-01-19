import functools
import inspect
import json

from joatmon.core import context
from joatmon.core.utility import (
    JSONEncoder,
    to_enumerable,
    to_hash
)


def cached(cache, duration):
    """
    Decorator for authorizing a function call.

    This decorator retrieves the current token and issuer from the context, and uses them to authorize the function call. If the authorization is successful, the function is called; otherwise, an exception is raised.

    Args:
        auth (str): The name of the authorizer in the context.
        token (str): The name of the token in the context.
        issuer (str): The name of the issuer in the context.

    Returns:
        function: The decorated function.
    """

    def _decorator(func):
        @functools.wraps(func)
        async def _wrapper(*args, **kwargs):
            cache_value = context.get_value(cache)

            key = to_hash(func, *args, **kwargs)
            if (value := await cache_value.get(key)) is None:
                result = await func(*args, **kwargs)
                await cache_value.add(key, json.dumps(to_enumerable(result), cls=JSONEncoder), duration)
            else:
                result = to_enumerable(json.loads(value))

            return result

        _wrapper.__signature__ = inspect.signature(func)
        return _wrapper

    return _decorator


def remove(cache, regex):
    """
    Decorator for removing entries from a cache.

    This decorator retrieves a cache from the context and uses it to remove entries that match a specified regular expression. After the entries are removed, the function is called.

    Args:
        cache (str): The name of the cache in the context.
        regex (str): The regular expression to match entries against.

    Returns:
        function: The decorated function.
    """

    def _decorator(func):
        @functools.wraps(func)
        async def _wrapper(*args, **kwargs):
            cache_value = context.get_value(cache)
            await cache_value.remove(regex)
            return await func(*args, **kwargs)

        _wrapper.__signature__ = inspect.signature(func)
        return _wrapper

    return _decorator
