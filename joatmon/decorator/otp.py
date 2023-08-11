import functools
import inspect

from joatmon import context


def authorized(auth, token, issuer):  # use current token and issuer
    def _decorator(func):
        @functools.wraps(func)
        async def _wrapper(*args, **kwargs):
            token_value = context.get_value(token).get()
            issuer_value = context.get_value(issuer).get()

            audience = f'{func.__module__}.{func.__qualname__}'

            authorizer = context.get_value(auth)
            await authorizer.authorize(token_value, issuer_value, audience)

            return await func(*args, **kwargs)

        _wrapper.__signature__ = inspect.signature(func)  # need to set min role here or maybe new decorator
        return _wrapper

    return _decorator
