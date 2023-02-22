import functools
import inspect

from core.response import PluginResponse, StatusCode
from core import ctx


def authorized(_func=None, min_role=None, auth=None, only_user=None):  # use current token and issuer
    def _decorator(func):
        @functools.wraps(func)
        async def _wrapper(*args, **kwargs):
            token = ctx.get_value('token')
            issuer = ctx.get_value('issuer')

            audience = f'{func.__module__}.{func.__qualname__}'

            authorizer = ctx.get_value(auth)
            auth_response = await authorizer.authorize(token, issuer, audience, min_role)
            if auth_response.code != StatusCode.OK:
                return auth_response

            user = auth_response.data
            ctx.set_value('user', user)

            if only_user and user.object_id != kwargs['user_id']:
                return PluginResponse(data=None, code=StatusCode.AUTH_NOT_ALLOWED)

            return await func(*args, **kwargs)

        _wrapper.__signature__ = inspect.signature(func)  # need to set min role here or maybe new decorator
        return _wrapper

    if _func is None:
        return _decorator
    else:
        return _decorator(_func)
