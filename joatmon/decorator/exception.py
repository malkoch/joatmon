import functools
import inspect


def handler(ex=None):
    """
    Decorator for handling exceptions in a function.

    This decorator wraps the function in a try-except block. If the function raises an exception of type `ex`, the exception is caught and its message is printed. The function then returns None.

    Args:
        ex (Exception, optional): The type of exception to catch. If None, all exceptions are caught. Defaults to None.

    Returns:
        function: The decorated function.
    """

    def _decorator(func):
        @functools.wraps(func)
        async def _wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ex as exception:
                print(str(exception))
                return None
            finally:
                return None

        _wrapper.__signature__ = inspect.signature(func)
        return _wrapper

    return _decorator
