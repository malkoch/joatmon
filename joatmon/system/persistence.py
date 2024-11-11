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
        return await self.db.insert(a, b)

    async def read(self, a, b):
        async for d in self.db.read(a, b):
            yield d

    async def update(self, a, b, c):
        return await self.db.update(a, b, c)

    async def delete(self, a, b):
        return await self.db.delete(a, b)
