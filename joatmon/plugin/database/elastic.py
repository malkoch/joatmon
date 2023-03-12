import uuid
from datetime import datetime

from elasticsearch import Elasticsearch

from joatmon import context
from joatmon.orm.document import Document
from joatmon.orm.meta import normalize_kwargs
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

        self.client.index(index=document.__metaclass__.__collection__, id=dictionary['object_id'], document=dictionary)

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
        result = self.client.get(index=document.__metaclass__.__collection__, document=normalize_kwargs(document.__metaclass__, **kwargs))

        for doc in result:
            yield document(**doc)

    async def update_raw(self, document):
        if not isinstance(document, Document):
            raise ValueError(f'{type(document)} is not valid for saving')

        dictionary = document.validate()

        self.client.update(index=document.__metaclass__.__collection__, id=dictionary['object_id'], document=dictionary)

        return document

    async def update(self, *documents):
        for document in documents:
            user = context.get_value(self.user_plugin).get()
            document.creator_id = user.object_id if user is not None else uuid.UUID(int=0)
            document.created_at = datetime.utcnow()
            document.updater_id = user.object_id if user is not None else uuid.UUID(int=0)
            document.updated_at = datetime.utcnow()

            await self.update_raw(document)

        return documents

    async def delete(self, *documents):
        for document in documents:
            user = context.get_value(self.user_plugin).get()
            document.updater_id = user.object_id if user is not None else uuid.UUID(int=0)
            document.updated_at = datetime.utcnow()
            document.deleter_id = user.object_id if user is not None else uuid.UUID(int=0)
            document.deleted_at = datetime.utcnow()
            document.is_deleted = True

            await self.update_raw(document)

        return documents

    async def start(self):
        ...

    async def commit(self):
        ...

    async def abort(self):
        ...

    async def end(self):
        ...
