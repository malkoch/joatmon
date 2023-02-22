import functools
import inspect
import re

from core import ctx
from core.utility import to_enumerable


def lang(_func=None, default='tr'):
    def _decorator(func):
        @functools.wraps(func)
        async def _wrapper(*args, **kwargs):
            language = ctx.get_value('lang')
            if language is None:
                language = default
            ctx.set_value('lang', language)
            result = await func(*args, **kwargs)
            return result

        _wrapper.__signature__ = inspect.signature(func)
        return _wrapper

    if _func is None:
        return _decorator
    else:
        return _decorator(_func)


async def helper(language, enumerable):
    if isinstance(enumerable, (list, tuple)):
        return [await helper(language, x) for x in enumerable]
    if isinstance(enumerable, dict):
        return {k: await helper(language, v) for k, v in enumerable.items()}
    if isinstance(enumerable, str):
        for x in re.findall(r'{@resource\.(.*?)}', enumerable):
            localised = await ctx.get_value('resource_service').get(x)
            enumerable = enumerable.replace('{@resource.' + x + '}', localised)
        return enumerable
    return enumerable


def localise(_func=None):
    def _decorator(func):
        @functools.wraps(func)
        async def _wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)

            language = ctx.get_value('lang')
            result = await helper(language, to_enumerable(result))

            return result

        _wrapper.__signature__ = inspect.signature(func)
        return _wrapper

    if _func is None:
        return _decorator
    else:
        return _decorator(_func)


localize = localise
