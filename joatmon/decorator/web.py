import functools
import inspect

from joatmon.core import context
from joatmon.core.exception import CoreException
from joatmon.core.utility import (
    to_case,
    to_enumerable
)


def get(func):
    """
    Decorator for HTTP GET method.

    This decorator marks the function as a handler for HTTP GET requests.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The decorated function.
    """
    func.__method__ = 'get'

    return func


def post(func):
    """
    Decorator for HTTP POST method.

    This decorator marks the function as a handler for HTTP POST requests.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The decorated function.
    """
    func.__method__ = 'post'

    return func


def incoming(case, json, arg, form):
    """
    Decorator for handling incoming requests.

    This decorator retrieves the request data from the context and updates the function's keyword arguments with it.

    Args:
        case (str): The name of the case in the context.
        json (str): The name of the JSON data in the context.
        arg (str): The name of the arguments in the context.
        form (str): The name of the form data in the context.

    Returns:
        function: The decorated function.
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
    Decorator for wrapping function calls.

    This decorator wraps the function call in a try-except block. If the function call is successful, it returns a dictionary with the result and a success status. If the function call raises a CoreException, it returns a dictionary with an error message and a failure status.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The decorated function.
    """

    @functools.wraps(func)
    async def _wrapper(*args, **kwargs):
        try:
            data = await func(*args, **kwargs)
            return {'data': to_enumerable(data), 'error': None, 'success': True}
        except CoreException as ex:
            return {'data': None, 'error': f'{{@resource.{ex}}}', 'success': False}

    _wrapper.__signature__ = inspect.signature(func)
    return _wrapper


def outgoing(case):
    """
    Decorator for handling outgoing responses.

    This decorator retrieves the case from the context and converts the function's return value to that case.

    Args:
        case (str): The name of the case in the context.

    Returns:
        function: The decorated function.
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
    Decorator for limiting requests per IP.

    This decorator retrieves the cache and the IP from the context. It limits the number of requests from the IP to one per specified interval.

    Args:
        interval (int): The interval in seconds between requests.
        cache (str): The name of the cache in the context.
        ip (str): The name of the IP in the context.

    Returns:
        function: The decorated function.
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
    Decorator for limiting requests.

    This decorator retrieves the cache from the context. It limits the number of requests to one per specified interval.

    Args:
        interval (int): The interval in seconds between requests.
        cache (str): The name of the cache in the context.

    Returns:
        function: The decorated function.
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
