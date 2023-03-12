import uuid
from datetime import datetime

from elasticsearch import Elasticsearch

from joatmon import context
from joatmon.orm.document import Document
from joatmon.plugin.database.core import DatabasePlugin


class ElasticDatabase(DatabasePlugin):
    DATABASES = set()
    CREATED_COLLECTIONS = set()
    UPDATED_COLLECTIONS = set()

    def __init__(self, uri, user_plugin):
        self.client = Elasticsearch(uri)
        self.user_plugin = user_plugin

    async def drop_database(self):
        ...

    async def drop_collection(self, collection):
        ...

    async def insert_raw(self, document):
        if not isinstance(document, Document):
            raise ValueError(f'{type(document)} is not valid for saving')

        dictionary = document.validate()

        self.client.index(index=document.__metaclass__.__collection__, document=dictionary)

        return document

    async def insert(self, *documents):
        for document in documents:
            user = context.get_value(self.user_plugin).get()
            document.creator_id = user.object_id if user is not None else uuid.UUID(int=0)
            document.created_at = datetime.utcnow()
            document.updater_id = user.object_id if user is not None else uuid.UUID(int=0)
            document.updated_at = datetime.utcnow()

            await self.insert_raw(document)

        return documents

    async def read(self, document, **kwargs):
        ...

    async def update_raw(self, document):
        ...

    async def update(self, *documents):
        ...

    async def delete(self, *documents):
        ...

    async def start(self):
        ...

    async def commit(self):
        ...

    async def abort(self):
        ...

    async def end(self):
        ...
