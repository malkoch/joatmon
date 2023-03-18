import uuid

from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from joatmon.core.utility import to_enumerable

from joatmon.plugin.database.core import DatabasePlugin


class CouchBaseDatabase(DatabasePlugin):
    DATABASES = set()
    CREATED_COLLECTIONS = set()
    UPDATED_COLLECTIONS = set()

    def __init__(self, uri, bucket, scope, username, password):
        auth = PasswordAuthenticator(username, password)

        self.bucket = bucket
        self.scope = scope

        self.db = Cluster(uri, ClusterOptions(auth)).bucket(bucket).scope(scope)

    async def _create_collection(self, collection):
        self.db.query(f'CREATE COLLECTION `{self.bucket}`.{self.scope}.{collection.__collection__} if not exits ')

    async def _get_collection(self, collection):
        return self.db.collection(collection)

    async def insert(self, document, *docs):
        for doc in docs:
            await self._create_collection(document.__metaclass__)

            self.db.collection(document.__metaclass__.__collection__).insert(str(uuid.uuid4()), to_enumerable(doc, string=True))

            yield document(**doc)

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
