import functools
import inspect


def retry(times=5):
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
            except Exception as ex:
                print(str(ex))
                return None
            finally:
                return None

        _wrapper.__signature__ = inspect.signature(func)
        return _wrapper

    return _decorator
