import redis

from joatmon.plugin.cache.core import CachePlugin


class RedisCache(CachePlugin):
    def __init__(self, host: str, port: int, password: str):
        self.host = host
        self.port = port
        self.password = password

        self.connection = redis.Redis(host=host, port=port, password=password)

    async def add(self, key, value, duration=None):
        self.connection.set(key, value, ex=duration)

    async def get(self, key):
        return self.connection.get(key)

    async def update(self, key, value):
        self.connection.set(key, value)

    async def remove(self, key):
        for k in self.connection.keys(key):
            self.connection.delete(k)
