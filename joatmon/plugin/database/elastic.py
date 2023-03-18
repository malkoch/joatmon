import warnings

from elasticsearch import Elasticsearch

from joatmon.plugin.database.core import DatabasePlugin


class ElasticDatabase(DatabasePlugin):
    DATABASES = set()
    CREATED_COLLECTIONS = set()
    UPDATED_COLLECTIONS = set()

    def __init__(self, uri):
        self.client = Elasticsearch(uri)

    async def insert(self, document, *docs):
        for doc in docs:
            if document.__metaclass__.structured:
                warnings.warn(f'document validation will be ignored')

            self.client.index(index=document.__metaclass__.__collection__, document=dict(**doc))

            yield doc

    async def read(self, document, query):
        raise NotImplementedError

    async def update(self, document, query, update):
        raise NotImplementedError

    async def delete(self, document, query):
        raise NotImplementedError

    async def start(self):
        raise NotImplementedError

    async def commit(self):
        raise NotImplementedError

    async def abort(self):
        raise NotImplementedError

    async def end(self):
        raise NotImplementedError
