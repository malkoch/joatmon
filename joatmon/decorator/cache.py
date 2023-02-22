import functools
import inspect
import json

from joatmon import context
from joatmon.core.utility import JSONEncoder, to_enumerable, to_hash
from joatmon.plugin.core import PluginResponse


def cached(_func=None, name=None, duration=None):
    def _decorator(func):
        @functools.wraps(func)
        async def _wrapper(*args, **kwargs):
            cache = context.get_value(name)

            key = f'{context.get_value("lang")}-{to_hash(func, *args, **kwargs)}'
            if (value := await cache.get(key)) is None:
                result = await func(*args, **kwargs)
                if isinstance(result, (DecoratorResponse, ServiceResponse, HTTPResponse, PluginResponse)):
                    result = result.data

                await cache.add(key, json.dumps(to_enumerable(result), cls=JSONEncoder), duration)
            else:
                result = to_enumerable(json.loads(value))

            return result

        _wrapper.__signature__ = inspect.signature(func)
        return _wrapper

    if _func is None:
        return _decorator
    else:
        return _decorator(_func)


def remove(_func=None, name=None, regex=None):
    def _decorator(func):
        @functools.wraps(func)
        async def _wrapper(*args, **kwargs):
            cache = context.get_value(name)
            await cache.remove(regex)
            return await func(*args, **kwargs)

        _wrapper.__signature__ = inspect.signature(func)
        return _wrapper

    if _func is None:
        return _decorator
    else:
        return _decorator(_func)
