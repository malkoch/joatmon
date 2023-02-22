import functools
import inspect

from joatmon import context
from joatmon.core.utility import to_case, to_enumerable
from joatmon.plugin.core import PluginResponse


def get(func):
    func.__method__ = 'get'

    return func


def post(func):
    func.__method__ = 'post'

    return func


def incoming(_func=None, case=None):
    def _decorator(func):
        @functools.wraps(func)
        async def _wrapper(*args, **kwargs):  # if func.method is get need to do with args, else with json
            c = context.get_value('in-case') or case

            if c is not None and context.get_value('json') is not None:
                kwargs.update(to_case(c, context.get_value('json')))
            if c is not None and context.get_value('args') is not None:
                kwargs.update(to_case(c, context.get_value('args')))
            if c is not None and context.get_value('form') is not None:
                kwargs.update(to_case(c, context.get_value('form')))

            return await func(*args, **kwargs)

        _wrapper.__signature__ = inspect.signature(func)
        return _wrapper

    if _func is None:
        return _decorator
    else:
        return _decorator(_func)


def wrap(func):
    @functools.wraps(func)
    async def _wrapper(*args, **kwargs):
        try:
            data = await func(*args, **kwargs)
            if isinstance(data, (ServiceResponse, PluginResponse, DecoratorResponse, HTTPResponse)):
                return HTTPResponse(to_enumerable(data.data), data.code).dict
            else:
                return HTTPResponse(to_enumerable(data), StatusCode.OK).dict
        except Exception as ex:
            print(str(ex))
            return HTTPResponse(None, StatusCode.UNKNOWN_ERROR).dict

    _wrapper.__signature__ = inspect.signature(func)
    return _wrapper


def outgoing(_func=None, case=None):
    def _decorator(func):
        @functools.wraps(func)
        async def _wrapper(*args, **kwargs):
            response = await func(*args, **kwargs)
            c = context.get_value('out-case') or case
            if c is not None:
                response = to_case(c, response)
            return response

        _wrapper.__signature__ = inspect.signature(func)
        return _wrapper

    if _func is None:
        return _decorator
    else:
        return _decorator(_func)


def ip_limit(_func=None, interval=1, cache=None):
    def _decorator(func):
        @functools.wraps(func)
        async def _wrapper(*args, **kwargs):
            c = context.get_value(cache)
            ip = context.get_value('ip')

            key = f'limit_request:{func.__qualname__}:{ip}'

            if await c.get(key) is not None:
                return DecoratorResponse(data=None, code=StatusCode.TOO_MANY_REQUEST)
            await c.add(key, 1, duration=interval)

            response = await func(*args, **kwargs)
            return response

        _wrapper.__signature__ = inspect.signature(func)
        return _wrapper

    if _func is None:
        return _decorator
    else:
        return _decorator(_func)


def limit(_func=None, interval=1, cache=None):
    def _decorator(func):
        @functools.wraps(func)
        async def _wrapper(*args, **kwargs):
            c = context.get_value(cache)

            key = f'limit_request:{func.__qualname__}'

            if await c.get(key) is not None:
                return DecoratorResponse(data=None, code=StatusCode.TOO_MANY_REQUEST)

            await c.add(key, 1, duration=interval)

            response = await func(*args, **kwargs)
            return response

        _wrapper.__signature__ = inspect.signature(func)
        return _wrapper

    if _func is None:
        return _decorator
    else:
        return _decorator(_func)
