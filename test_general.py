import asyncio
import datetime

from joatmon import context
from joatmon.plugin.auth.jwt import JWTAuth
from joatmon.plugin.cache.redis import RedisCache
from joatmon.plugin.core import register


class CTX:
    ...


ctx = CTX()
context.set_ctx(ctx)

register(JWTAuth, 'auth', 'secretkeysecretkeysecretkeysecretkeysecretkeysecretkey')
register(RedisCache, 'redis_cache', '127.0.0.1', 6379, 'malkoch')


async def test():
    token = await context.get_value('auth').authenticate(1, 'hello', ['world'], datetime.datetime.now() + datetime.timedelta(hours=1))
    print(token)
    response = await context.get_value('auth').authorize(token, 'hello', 'world', 1)
    print(response)

    await context.get_value('redis_cache').add('k', 'v', duration=1)
    print(await context.get_value('redis_cache').get('k'))
    await asyncio.sleep(1)
    print(await context.get_value('redis_cache').get('k'))


asyncio.run(test())
