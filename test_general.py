import asyncio
import datetime

from joatmon import context
from joatmon.plugin.auth.jwt import JWTAuth
from joatmon.plugin.core import register


class CTX:
    ...


ctx = CTX()
context.set_ctx(ctx)

register(JWTAuth, 'auth', 'secretkeysecretkeysecretkeysecretkeysecretkeysecretkey')


async def test():
    token = await context.get_value('auth').authenticate(1, 'hello', ['world'], datetime.datetime.now() + datetime.timedelta(hours=1))
    print(token)
    response = await context.get_value('auth').authorize(token, 'hello', 'world', 1)
    print(response)


asyncio.run(test())
