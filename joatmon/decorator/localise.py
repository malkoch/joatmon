import functools
import inspect
import re

from joatmon import context
from joatmon.core.utility import to_enumerable


async def helper(language, enumerable):
    if isinstance(enumerable, (list, tuple)):
        return [await helper(language, x) for x in enumerable]
    if isinstance(enumerable, dict):
        return {k: await helper(language, v) for k, v in enumerable.items()}
    if isinstance(enumerable, str):
        for x in re.findall(r'{@resource\.(.*?)}', enumerable):
            localised = await context.get_value('resource_service').get(x)
            enumerable = enumerable.replace('{@resource.' + x + '}', localised)
        return enumerable
    return enumerable


def localise(language):
    def _decorator(func):
        @functools.wraps(func)
        async def _wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)

            language_value = context.get_value(language).get()
            result = await helper(language_value, to_enumerable(result))

            return result

        _wrapper.__signature__ = inspect.signature(func)
        return _wrapper

    return _decorator


localize = localise
