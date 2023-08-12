import functools
import warnings
from typing import Callable


def deprecated(reason: str) -> Callable:
    """
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    """

    def decorator(func1):
        message = 'Call to deprecated function {name} ({reason}).'

        @functools.wraps(func1)
        def new_func1(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn(
                message.format(name=func1.__name__, reason=reason),
                category=DeprecationWarning,
                stacklevel=2,
            )
            warnings.simplefilter('default', DeprecationWarning)
            return func1(*args, **kwargs)

        return new_func1

    return decorator
