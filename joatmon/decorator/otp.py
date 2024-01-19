import functools
import inspect

from joatmon.core import context


def authorized(auth, token, issuer):  # use current token and issuer
    """
    Decorator for authorizing function calls.

    This decorator retrieves an authorizer, a token, and an issuer from the context. It uses the authorizer to authorize the function call with the token, the issuer, and the audience. The audience is the fully qualified name of the function.

    Args:
        auth (str): The name of the authorizer in the context.
        token (str): The name of the token in the context.
        issuer (str): The name of the issuer in the context.

    Returns:
        function: The decorated function.
    """

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
