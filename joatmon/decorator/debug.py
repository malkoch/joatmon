import functools
import inspect
import time

from joatmon import context


def timeit():
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
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
