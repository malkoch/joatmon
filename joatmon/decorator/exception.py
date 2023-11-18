import functools
import inspect


def handler(ex=None):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
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
