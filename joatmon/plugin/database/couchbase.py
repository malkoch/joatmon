import uuid
from datetime import datetime

from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions

from joatmon import context
from joatmon.core.utility import to_enumerable
from joatmon.orm.document import Document
from joatmon.orm.meta import normalize_kwargs
from joatmon.plugin.database.core import DatabasePlugin


class CouchBaseDatabase(DatabasePlugin):
    DATABASES = set()
    CREATED_COLLECTIONS = set()
    UPDATED_COLLECTIONS = set()

    def __init__(self, uri, bucket, scope, username, password, user_plugin):
        self.user_plugin = user_plugin

        auth = PasswordAuthenticator(username, password)
        cluster = Cluster(uri, ClusterOptions(auth))
        bucket = cluster.bucket(bucket)
        scope = bucket.scope(scope)

        self.bucket = bucket
        self.scope = scope

        self.db = scope

    async def _check_collection(self, collection):
        return False

    async def _create_collection(self, collection):
        self.db.query(f'CREATE COLLECTION `{self.bucket}`.{self.scope}.{collection.__collection__} if not exits ')

    async def _ensure_collection(self, collection):
        if not await self._check_collection(collection):
            await self._create_collection(collection)

    async def _get_collection(self, collection):
        return self.db.collection(collection)

    async def drop_database(self):
        ...

    async def drop_collection(self, collection):
        ...

    async def insert_raw(self, document):
        if not isinstance(document, Document):
            raise ValueError(f'{type(document)} is not valid for saving')

        await self._ensure_collection(document.__metaclass__)

        dictionary = document.validate()

        collection = await self._get_collection(document.__metaclass__.__collection__)
        collection.insert(str(dictionary['object_id']), to_enumerable(dictionary, string=True))

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
        await self._ensure_collection(document.__metaclass__)

        collection = await self._get_collection(document.__metaclass__.__collection__)
        result = collection.get(normalize_kwargs(document.__metaclass__, **kwargs))

        for doc in result:
            yield document(**doc)

    async def update_raw(self, document):
        if not isinstance(document, Document):
            raise ValueError(f'{type(document)} is not valid for saving')

        dictionary = document.validate()

        query = {'object_id': document.object_id}
        update = {'$set': dictionary}

        await self._ensure_collection(document.__metaclass__)
        collection = await self._get_collection(document.__metaclass__.__collection__)
        collection.upsert(query, update, session=self.session)

        return document

    async def update(self, *documents):
        for document in documents:
            user = context.get_value(self.user_plugin).get()
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
