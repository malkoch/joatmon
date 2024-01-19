import functools
import inspect
import warnings
from typing import Callable


def deprecated(reason: str) -> Callable:
    """
    Decorator for marking functions as deprecated.

    This decorator emits a warning when the decorated function is called. The warning includes the name of the function and the reason for its deprecation.

    Args:
        reason (str): The reason for the function's deprecation.

    Returns:
        Callable: The decorated function.
    """

    def _decorator(func):
        message = 'Call to deprecated function {name} ({reason}).'

        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn(
                f'Call to deprecated function {func.__name__} ({reason}).',
                category=DeprecationWarning,
                stacklevel=2,
            )
            warnings.simplefilter('default', DeprecationWarning)
            return func(*args, **kwargs)

        _wrapper.__signature__ = inspect.signature(func)
        return _wrapper

    return _decorator
