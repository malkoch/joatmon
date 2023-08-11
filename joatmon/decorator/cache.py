import functools
import inspect
import json

from joatmon import context
from joatmon.utility import (
    JSONEncoder,
    to_enumerable,
    to_hash
)


def cached(cache, duration):
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
    def _decorator(func):
        @functools.wraps(func)
        async def _wrapper(*args, **kwargs):
            cache_value = context.get_value(cache)
            await cache_value.remove(regex)
            return await func(*args, **kwargs)

        _wrapper.__signature__ = inspect.signature(func)
        return _wrapper

    return _decorator
