import functools
import inspect
import os
import sys
import time
import traceback

from joatmon import context
from joatmon.core.utility import to_enumerable


def log(logger, on_begin=None, on_success=None, on_error=None):
    # need to stop writing self values
    async def _on_begin(f, *args, **kwargs):
        logger_value = context.get_value(logger)

        await logger_value.info(
            {
                'function': f.__qualname__,
                'module': f.__module__,
                'args': to_enumerable(args, string=True),
                'kwargs': to_enumerable(kwargs, string=True)
            }
        )

    async def _on_success(f, t, result, *args, **kwargs):
        logger_value = context.get_value(logger)

        await logger_value.info(
            {
                'timed': t,
                'result': to_enumerable(result, string=True),
                'function': f.__qualname__,
                'module': f.__module__,
                'args': to_enumerable(args, string=True),
                'kwargs': to_enumerable(kwargs, string=True)
            }
        )

    async def _on_error(f, exception, *args, **kwargs):
        logger_value = context.get_value(logger)

        exc_type, exc_obj, exc_trace = sys.exc_info()

        await logger_value.error(
            {
                'exception': {
                    'line': exc_trace.tb_lineno,
                    'file': os.path.split(exc_trace.tb_frame.f_code.co_filename)[1],
                    'message': str(exception),
                    'trace': traceback.format_exc()
                },
                'function': f.__qualname__,
                'module': f.__module__,
                'args': to_enumerable(args, string=True),
                'kwargs': to_enumerable(kwargs, string=True)
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

    return _decorator
