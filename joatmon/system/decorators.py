import functools
import inspect
import time

__all__ = ['auto_pause']


def auto_pause(duration):
    def _decorator(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            ret = func(*args, **kwargs)
            time.sleep(duration)
            return ret

        _wrapper.__signature__ = inspect.signature(func)  # need to set min role here or maybe new decorator
        return _wrapper

    return _decorator
