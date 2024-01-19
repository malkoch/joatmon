import functools
import inspect
import time


def timeit():
    """
    Decorator for timing the execution of a function.

    This decorator measures the time it takes to execute the decorated function. It prints the function's name and the execution time in seconds.

    Returns:
        function: The decorated function.
    """

    def _decorator(func):
        @functools.wraps(func)
        async def _wrapper(*args, **kwargs):
            begin = time.time()
            ret = await func(*args, **kwargs)
            end = time.time()

            print(f'{func.__name__} took {end - begin} seconds to run')

            return ret

        _wrapper.__signature__ = inspect.signature(func)  # need to set min role here or maybe new decorator
        return _wrapper

    return _decorator
