from joatmon.plugin.cache.core import Cache


class MemoryCache(Cache):
    def __init__(self, alias: str):
        super(MemoryCache, self).__init__(alias)

        self._cache = {}

    def remove_all(self):
        self._cache = {}

    def add(self, key, value):
        self._cache[key] = value

    def update(self, key, value):
        self._cache[key] = value

    def remove(self, key):
        del self._cache[key]

    def get(self, key):
        return self._cache[key]

    def has(self, key):
        return key in self._cache
