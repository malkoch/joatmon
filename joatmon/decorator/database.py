import functools
import inspect

import async_exit_stack

from joatmon.core import context


def transaction(names):
    """
    Decorator for managing database transactions.

    This decorator retrieves one or more database connections from the context and uses them to manage a database transaction. The transaction is started before the function is called and is committed or rolled back after the function is called, depending on whether the function raises an exception.

    Args:
        names (list of str): The names of the database connections in the context.

    Returns:
        function: The decorated function.
    """

    def _decorator(func):
        @functools.wraps(func)
        async def _wrapper(*args, **kwargs):
            async with async_exit_stack.AsyncExitStack() as stack:
                for name in names or []:
                    await stack.enter_async_context(context.get_value(name))

                return await func(*args, **kwargs)

        _wrapper.__signature__ = inspect.signature(func)
        return _wrapper

    return _decorator
