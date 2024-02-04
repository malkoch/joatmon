import functools
import inspect
import time

__all__ = ['auto_pause']


def auto_pause(duration):
    """
    Decorator to pause the execution of a function for a specified duration.

    Args:
        duration (float): The duration to pause the function execution in seconds.

    Returns:
        function: The decorated function which will pause for the specified duration after execution.
    """

    def _decorator(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            ret = func(*args, **kwargs)
            time.sleep(duration)
            return ret

        _wrapper.__signature__ = inspect.signature(func)  # need to set min role here or maybe new decorator
        return _wrapper

    return _decorator
