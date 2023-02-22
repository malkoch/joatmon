from joatmon.plugin.core import Plugin


class DatabasePlugin(Plugin):
    async def __aenter__(self):
        await self.start()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            await self.commit()
        else:
            await self.abort()
            raise exc_val
        await self.end()

        return self

    async def drop_database(self):
        ...

    async def drop_collection(self, collection):
        ...

    async def insert_raw(self, document):
        ...

    async def insert(self, *documents):
        ...

    async def read(self, document):
        ...

    async def update_raw(self, document):
        ...

    async def update(self, *documents):
        ...

    async def delete(self):
        ...

    async def start(self):
        ...

    async def commit(self):
        ...

    async def abort(self):
        ...

    async def end(self):
        ...
