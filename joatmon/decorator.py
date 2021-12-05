import inspect
import os
import sys
import traceback
from functools import wraps
from time import (
    sleep,
    time
)

from joatmon.context import current
from joatmon.core import CoreException
from joatmon.plugin.core import create
from joatmon.utility import (
    to_case,
    to_enumerable,
    to_hash
)
from joatmon.web.response import Response


def retry():
    ...


def debug(func):
    @wraps(func)
    def _wrapper(*args, **kwargs):
        params = f'args: {args}, kwargs: {kwargs}'
        begin = time()
        result = func(*args, **kwargs)
        end = time()
        print(f'function {func.__name__} is called with params {params} and resulted with {result} in {end - begin} seconds.')
        return result

    return _wrapper


def pause(_func=None, timeout=None):
    def _decorator(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            sleep(timeout)
            response = func(*args, **kwargs)
            return response

        _wrapper.__signature__ = inspect.signature(func)
        return _wrapper

    if _func is None:
        return _decorator
    else:
        return _decorator(_func)


def private(func):
    @wraps(func)
    def _wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    _wrapper.access = 'private'
    _wrapper.__signature__ = inspect.signature(func)
    return _wrapper


def protected(func):
    @wraps(func)
    def _wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    _wrapper.access = 'protected'
    _wrapper.__signature__ = inspect.signature(func)
    return _wrapper


def public(func):
    @wraps(func)
    def _wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    _wrapper.access = 'public'
    _wrapper.__signature__ = inspect.signature(func)
    return _wrapper


def post(func):
    @wraps(func)
    def _wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    _wrapper.method = 'post'
    _wrapper.__signature__ = inspect.signature(func)
    return _wrapper


def get(func):
    @wraps(func)
    def _wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    _wrapper.method = 'get'
    _wrapper.__signature__ = inspect.signature(func)
    return _wrapper


def put(func):
    @wraps(func)
    def _wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    _wrapper.method = 'put'
    _wrapper.__signature__ = inspect.signature(func)
    return _wrapper


def delete(func):
    @wraps(func)
    def _wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    _wrapper.method = 'delete'
    _wrapper.__signature__ = inspect.signature(func)
    return _wrapper


def lang(_func=None, default='tr'):
    def _decorator(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            language = current.get('headers', {}).get('Language', None)
            if language is None:
                language = default
            current['language'] = language
            result = func(*args, **kwargs)
            return result

        _wrapper.__signature__ = inspect.signature(func)
        return _wrapper

    if _func is None:
        return _decorator
    else:
        return _decorator(_func)


def log(_func=None, name=None, on_begin=None, on_success=None, on_error=None):
    # need to stop writing self values
    def _on_begin(f, *args, **kwargs):
        logger = create(name)

        logger.info({
            'ip': current.get('ip', None),
            'function': f.__qualname__,
            'module': f.__module__,
            'args': to_enumerable(args, string=True),
            'kwargs': to_enumerable(kwargs, string=True),
            'language': current.get('language', None)
        })

    def _on_success(f, t, result, *args, **kwargs):
        logger = create(name)

        logger.info({
            'timed': t,
            'ip': current.get('ip', None),
            'result': to_enumerable(result, string=True),
            'function': f.__qualname__,
            'module': f.__module__,
            'args': to_enumerable(args, string=True),
            'kwargs': to_enumerable(kwargs, string=True),
            'language': current.get('language', None)
        })

    def _on_error(f, exception, *args, **kwargs):
        logger = create(name)

        exc_type, exc_obj, exc_trace = sys.exc_info()

        logger.error({
            'ip': current.get('ip', None),
            'exception': {
                'line': exc_trace.tb_lineno,
                'file': os.path.split(exc_trace.tb_frame.f_code.co_filename)[1],
                'message': str(exception),
                'trace': traceback.format_exc()
            },
            'function': f.__qualname__,
            'module': f.__module__,
            'args': to_enumerable(args, string=True),
            'kwargs': to_enumerable(kwargs, string=True),
            'language': current.get('language', None)
        })

    on_begin = on_begin or _on_begin
    on_success = on_success or _on_success
    on_error = on_error or _on_error

    def _decorator(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            on_begin(func, *args, **kwargs)  # need to make sure these function does not raises exception
            try:

                begin = time()
                result = func(*args, **kwargs)
                end = time()

                on_success(func, end - begin, result, *args, **kwargs)  # need to make sure these function does not raises exception

                return result
            except Exception as ex:
                on_error(func, ex, *args, **kwargs)  # need to make sure these function does not raises exception
                raise ex

        _wrapper.__signature__ = inspect.signature(func)
        return _wrapper

    if _func is None:
        return _decorator
    else:
        return _decorator(_func)


def localise(_func=None, default=None, database=None):
    ...


def outgoing(_func=None, case=None):
    def _decorator(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            response = func(*args, **kwargs)
            c = current.get('headers', {}).get('out-case') or case
            if c is not None:
                response = to_case(c, response)
            return response

        _wrapper.__signature__ = inspect.signature(func)
        return _wrapper

    if _func is None:
        return _decorator
    else:
        return _decorator(_func)


def wrap(func):
    @wraps(func)
    def _wrapper(*args, **kwargs):
        try:
            data = func(*args, **kwargs)
            return Response(to_enumerable(data), True).dict
        except Exception as ex:
            if isinstance(ex, CoreException):
                data, success, msg = None, False, str(ex)
            else:
                data, success, msg = None, False, 'error'
            return Response(to_enumerable(data), success, msg).dict

    _wrapper.__signature__ = inspect.signature(func)
    return _wrapper


def incoming(_func=None, case=None):
    def _decorator(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):  # if func.method is get need to do with args, else with json
            c = current.get('headers', {}).get('in-case', None) or case

            if c is not None and current.get('json', None) is not None:
                kwargs.update(to_case(c, current.get('json', None)))
            if c is not None and current.get('args', None) is not None:
                kwargs.update(to_case(c, current.get('args', None)))
            if c is not None and current.get('form', None) is not None:
                kwargs.update(to_case(c, current.get('form', None)))

            return func(*args, **kwargs)

        _wrapper.__signature__ = inspect.signature(func)
        return _wrapper

    if _func is None:
        return _decorator
    else:
        return _decorator(_func)


def session(_func=None, database=None):
    def _decorator(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            with create(database):
                return func(*args, **kwargs)

        _wrapper.__signature__ = inspect.signature(func)
        return _wrapper

    if _func is None:
        return _decorator
    else:
        return _decorator(_func)


def authorized(_func=None, min_role=None, auth=None):  # use current token and issuer
    def _decorator(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            token = current.get('token', None)
            if token is None:
                raise CoreException('token_not_found')
            issuer = current.get('issuer', None)
            if issuer is None:
                raise CoreException('issuer_not_found')

            audience = f'{func.__module__}.{func.__qualname__}'

            authorizer = create(auth)
            user = authorizer.authorize(token, issuer, audience, min_role)
            current['user'] = user

            return func(*args, **kwargs)

        _wrapper.__signature__ = inspect.signature(func)  # need to set min role here or maybe new decorator
        return _wrapper

    if _func is None:
        return _decorator
    else:
        return _decorator(_func)


def cached(_func=None, name=None):
    def _decorator(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            cache = create(name)

            if cache is not None and 'get' in func.__name__:
                key = to_hash(func, *args, **kwargs)
                if not cache.has(key):
                    result = func(*args, **kwargs)
                    cache.add(key, result)
                result = cache.get(key)
            else:
                result = func(*args, **kwargs)

            return result

        _wrapper.__signature__ = inspect.signature(func)
        return _wrapper

    if _func is None:
        return _decorator
    else:
        return _decorator(_func)
