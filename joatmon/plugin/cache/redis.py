import json

import redis

from joatmon.plugin.cache.core import Cache
from joatmon.utility import JSONEncoder


class RedisCache(Cache):
    def __init__(self, alias: str, connection: str):
        super(RedisCache, self).__init__(alias)

        self._cache = redis.Redis(connection_pool=redis.BlockingConnectionPool.from_url(connection))

    def remove_all(self):
        self._cache.flushall()

    def add(self, key, value):
        if hasattr(value, 'dict'):
            value = json.dumps(value.dict, cls=JSONEncoder)
        value = value.encode('utf-8')
        self._cache.set(key, value)

    def update(self, key, value):
        self._cache.set(key, value)

    def remove(self, key):
        self._cache.delete(key)

    def get(self, key):
        value = self._cache.get(key)
        value = value.decode('utf-8')
        try:
            value = json.loads(value)
        except:
            ...
        return value

    def has(self, key):
        return self._cache.exists(key)
