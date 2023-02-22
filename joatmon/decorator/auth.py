import functools
import inspect

from joatmon import context
from joatmon.plugin.auth.core import AuthStatusCode


def authorized(_func=None, min_role=None, auth=None):  # use current token and issuer
    def _decorator(func):
        @functools.wraps(func)
        async def _wrapper(*args, **kwargs):
            token = context.get_value('token')
            issuer = context.get_value('issuer')

            audience = f'{func.__module__}.{func.__qualname__}'

            authorizer = context.get_value(auth)
            auth_response = await authorizer.authorize(token, issuer, audience, min_role)
            if auth_response.code != AuthStatusCode.OK:
                return auth_response

            return await func(*args, **kwargs)

        _wrapper.__signature__ = inspect.signature(func)  # need to set min role here or maybe new decorator
        return _wrapper

    if _func is None:
        return _decorator
    else:
        return _decorator(_func)
