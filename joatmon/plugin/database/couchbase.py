import uuid

from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.logic.n1ql import QueryScanConsistency
from couchbase.options import (
    ClusterOptions,
    QueryOptions
)

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

            values = '{' + ', '.join([f'"{k}" : ${k}' for k, v in doc.items()]) + '}'
            self.db.query(
                f'insert into `{self.bucket}`.{self.scope}.{document.__metaclass__.__collection__} '
                f'(key, value) '
                f'values '
                f'(\'{str(uuid.uuid4())}\', {values}) '
                f'returning *',
                QueryOptions(named_parameters={k: str(v) if isinstance(v, uuid.UUID) else v for k, v in doc.items()}, scan_consistency=QueryScanConsistency.REQUEST_PLUS)
            ).execute()

            yield document(**doc)

    async def read(self, document, query):
        await self._create_collection(document.__metaclass__)

        values = 'and '.join([f'`{k}` = ${k}' for k, v in query.items()])
        result = self.db.query(
            f'SELECT * '
            f'FROM `{self.bucket}`.{self.scope}.{document.__metaclass__.__collection__} '
            f'WHERE {values}',
            QueryOptions(named_parameters={k: str(v) if isinstance(v, uuid.UUID) else v for k, v in query.items()}, scan_consistency=QueryScanConsistency.REQUEST_PLUS)
        ).execute()
        for r in result:
            yield document(**r[document.__metaclass__.__collection__])

    async def update(self, document, query, update):
        query_values = 'and '.join([f'`{k}` = $query_{k}' for k, v in query.items()])
        update_values = ', '.join([f'`{k}` = $update_{k}' for k, v in update.items()])

        params = {}
        params.update({f'query_{k}': str(v) if isinstance(v, uuid.UUID) else v for k, v in query.items()})
        params.update({f'update_{k}': str(v) if isinstance(v, uuid.UUID) else v for k, v in update.items()})
        self.db.query(
            f'update `{self.bucket}`.{self.scope}.{document.__metaclass__.__collection__} '
            f'set {update_values} '
            f'WHERE {query_values}',
            QueryOptions(named_parameters=params, scan_consistency=QueryScanConsistency.REQUEST_PLUS)
        ).execute()

    async def delete(self, document, query):
        values = 'and '.join([f'`{k}` = ${k}' for k, v in query.items()])
        self.db.query(
            f'delete from `{self.bucket}`.{self.scope}.{document.__metaclass__.__collection__} '
            f'WHERE {values}',
            QueryOptions(named_parameters={k: str(v) if isinstance(v, uuid.UUID) else v for k, v in query.items()}, scan_consistency=QueryScanConsistency.REQUEST_PLUS)
        ).execute()

    async def start(self):
        raise NotImplementedError

    async def commit(self):
        raise NotImplementedError

    async def abort(self):
        raise NotImplementedError

    async def end(self):
        raise NotImplementedError
