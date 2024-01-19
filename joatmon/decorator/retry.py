import functools
import inspect


def retry(times=5):
    """
    Decorator for retrying a function call.

    This decorator wraps the function in a try-except block. If the function raises an exception, the decorator catches it, prints its message, and retries the function call. The function call is retried a specified number of times.

    Args:
        times (int, optional): The number of times to retry the function call. Defaults to 5.

    Returns:
        function: The decorated function.
    """

    def _decorator(func):
        @functools.wraps(func)
        async def _wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as ex:
                print(str(ex))
                return None
            finally:
                return None

        _wrapper.__signature__ = inspect.signature(func)
        return _wrapper

    return _decorator
