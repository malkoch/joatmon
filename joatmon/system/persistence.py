from joatmon.core import context
from joatmon.core.exception import CoreException
from joatmon.system.module import Module


class PersistenceException(CoreException):
    ...


class PersistenceModule(Module):
    def __init__(self, system, plugin):
        super().__init__(system)

        self.db = context.get_value(plugin)

    async def insert(self, doc, insert):
        return await self.db.insert(doc, insert)

    async def read(self, doc, query):
        async for d in self.db.read(doc, query):
            yield d

    async def update(self, doc, query, update):
        return await self.db.update(doc, query, update)

    async def delete(self, doc, query):
        return await self.db.delete(doc, query)
