from joatmon.core import context
from joatmon.core.exception import CoreException
from joatmon.system.module import Module


class PersistenceException(CoreException):
    ...


class PersistenceModule(Module):
    def __init__(self, system, plugin):
        super().__init__(system)

        self.db = context.get_value(plugin)

    async def insert(self, a, b):
        return self.db.insert(a, b)

    async def read(self, a, b):
        return self.db.read(a, b)

    async def update(self, a, b, c):
        return self.db.update(a, b, c)

    async def delete(self, a, b):
        return self.db.delete(a, b)
