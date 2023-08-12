import functools
import inspect

from joatmon import context
from joatmon.utility import (
    to_case,
    to_enumerable
)


def get(func):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    func.__method__ = 'get'

    return func


def post(func):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    func.__method__ = 'post'

    return func


def incoming(case, json, arg, form):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """

    def _decorator(func):
        @functools.wraps(func)
        async def _wrapper(*args, **kwargs):  # if func.method is get need to do with args, else with json
            c = context.get_value(case).get()
            c = 'snake'

            if (j := context.get_value(json).get()) is not None:
                if c is not None:
                    kwargs.update(to_case(c, j))
                else:
                    kwargs.update(j)

            if (a := context.get_value(arg).get()) is not None:
                if c is not None:
                    kwargs.update(to_case(c, a))
                else:
                    kwargs.update(a)

            if (f := context.get_value(form).get()) is not None:
                if c is not None:
                    kwargs.update(to_case(c, f))
                else:
                    kwargs.update(f)

            return await func(*args, **kwargs)

        _wrapper.__signature__ = inspect.signature(func)
        return _wrapper

    return _decorator


def wrap(func):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """

    @functools.wraps(func)
    async def _wrapper(*args, **kwargs):
        try:
            data = await func(*args, **kwargs)
            return {'data': to_enumerable(data), 'error': None, 'success': True}
        except Exception as ex:
            return {'data': None, 'error': str(ex), 'success': False}

    _wrapper.__signature__ = inspect.signature(func)
    return _wrapper


def outgoing(case):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """

    def _decorator(func):
        @functools.wraps(func)
        async def _wrapper(*args, **kwargs):
            response = await func(*args, **kwargs)
            c = context.get_value(case).get()
            c = 'pascal'
            if c is not None:
                response = to_case(c, response)
            return response

        _wrapper.__signature__ = inspect.signature(func)
        return _wrapper

    return _decorator


def ip_limit(interval, cache, ip):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """

    def _decorator(func):
        @functools.wraps(func)
        async def _wrapper(*args, **kwargs):
            c = context.get_value(cache)
            ip_value = context.get_value(ip).get()

            key = f'limit_request:{func.__qualname__}:{ip_value}'

            if await c.get(key) is not None:
                raise ValueError('too_many_request')
            await c.add(key, 1, duration=interval)

            response = await func(*args, **kwargs)
            return response

        _wrapper.__signature__ = inspect.signature(func)
        return _wrapper

    return _decorator


def limit(interval, cache):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """

    def _decorator(func):
        @functools.wraps(func)
        async def _wrapper(*args, **kwargs):
            c = context.get_value(cache)

            key = f'limit_request:{func.__qualname__}'

            if await c.get(key) is not None:
                raise ValueError('too_many_request')

            await c.add(key, 1, duration=interval)

            response = await func(*args, **kwargs)
            return response

        _wrapper.__signature__ = inspect.signature(func)
        return _wrapper

    return _decorator
