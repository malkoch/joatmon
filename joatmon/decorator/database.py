import functools
import inspect

import async_exit_stack

from joatmon import context


def transaction(names):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
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
