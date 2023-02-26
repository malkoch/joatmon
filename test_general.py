import asyncio
import functools
import inspect
import re

from joatmon.core.utility import to_enumerable


async def helper(enumerable, pattern):
    if isinstance(enumerable, (list, tuple)):
        return [await helper(x, pattern) for x in enumerable]
    if isinstance(enumerable, dict):
        return {k: await helper(v, pattern) for k, v in enumerable.items()}
    if isinstance(enumerable, str):
        for x in re.findall(pattern, enumerable):
            # localised = await context.get_value('resource_service').get(x)
            print(enumerable, end=' ')
            enumerable = enumerable.replace('{@resource.' + x + '}', x)
            print(enumerable)
        return enumerable
    return enumerable


def localise(language, pattern):
    def _decorator(func):
        @functools.wraps(func)
        async def _wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)

            result = await helper(to_enumerable(result), pattern)

            return result

        _wrapper.__signature__ = inspect.signature(func)
        return _wrapper

    return _decorator


@localise('tr', r'{@resource\.(.*?)}')
async def aaa():
    return {'hello': '{@resource.hello}'}


asyncio.run(aaa())
