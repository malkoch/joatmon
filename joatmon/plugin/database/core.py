from joatmon.plugin.core import Plugin


class DatabasePlugin(Plugin):
    # query can be a dictionary, query builder or formatted query
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

    async def create(self, document):
        raise NotImplementedError

    async def alter(self, document):
        raise NotImplementedError

    async def drop(self, document):
        raise NotImplementedError

    async def insert(self, document, *docs):
        raise NotImplementedError

    async def read(self, document, query):
        raise NotImplementedError

    async def update(self, document, query, update):
        raise NotImplementedError

    async def delete(self, document, query):
        raise NotImplementedError

    async def view(self, document, query):
        raise NotImplementedError

    async def execute(self, document, query):
        raise NotImplementedError

    async def count(self, query):
        raise NotImplementedError

    async def start(self):
        raise NotImplementedError

    async def commit(self):
        raise NotImplementedError

    async def abort(self):
        raise NotImplementedError

    async def end(self):
        raise NotImplementedError
