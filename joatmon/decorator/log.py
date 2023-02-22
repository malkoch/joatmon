import functools
import inspect
import os
import sys
import time
import traceback

from core import ctx
from core.utility import to_enumerable


def log(_func=None, name=None, on_begin=None, on_success=None, on_error=None):
    # need to stop writing self values
    async def _on_begin(f, *args, **kwargs):
        logger = ctx.get_value(name)

        await logger.info(
            {
                'ip': ctx.get_value('ip'),
                'function': f.__qualname__,
                'module': f.__module__,
                'args': to_enumerable(args, string=True),
                'kwargs': to_enumerable(kwargs, string=True),
                'language': ctx.get_value('lang')
            }
        )

    async def _on_success(f, t, result, *args, **kwargs):
        logger = ctx.get_value(name)

        await logger.info(
            {
                'timed': t,
                'ip': ctx.get_value('ip'),
                'result': to_enumerable(result, string=True),
                'function': f.__qualname__,
                'module': f.__module__,
                'args': to_enumerable(args, string=True),
                'kwargs': to_enumerable(kwargs, string=True),
                'language': ctx.get_value('lang')
            }
        )

    async def _on_error(f, exception, *args, **kwargs):
        logger = ctx.get_value(name)

        exc_type, exc_obj, exc_trace = sys.exc_info()

        await logger.error(
            {
                'ip': ctx.get_value('ip'),
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
                'language': ctx.get_value('lang')
            }
        )

    on_begin = on_begin or _on_begin
    on_success = on_success or _on_success
    on_error = on_error or _on_error

    def _decorator(func):
        @functools.wraps(func)
        async def _wrapper(*args, **kwargs):
            await on_begin(func, *args, **kwargs)
            try:
                begin = time.time()
                result = await func(*args, **kwargs)
                end = time.time()

                await on_success(func, end - begin, result, *args, **kwargs)

                return result
            except Exception as ex:
                await on_error(func, ex, *args, **kwargs)
                raise ex

        _wrapper.__signature__ = inspect.signature(func)
        return _wrapper

    if _func is None:
        return _decorator
    else:
        return _decorator(_func)
