from joatmon.plugin.core import Plugin


class CachePlugin(Plugin):
    async def add(self, key, value, duration=None):
        raise NotImplementedError

    async def get(self, key):
        raise NotImplementedError

    async def update(self, key, value):
        raise NotImplementedError

    async def remove(self, key):
        raise NotImplementedError
