from joatmon.core import CoreException
from joatmon.plugin.core import Plugin


class CacheException(CoreException):
    ...


class KeyNotFoundError(CacheException):
    pass


class KeyAlreadyExists(CacheException):
    pass


class Cache(Plugin):
    def __init__(self, alias: str):
        super(Cache, self).__init__(alias)

    def remove_all(self):
        raise NotImplementedError

    def add(self, key, value):
        raise NotImplementedError

    def update(self, key, value):
        raise NotImplementedError

    def remove(self, key):
        raise NotImplementedError

    def get(self, key):
        raise NotImplementedError

    def has(self, key):
        raise NotImplementedError
