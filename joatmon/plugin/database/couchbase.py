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
    """
    CouchBaseDatabase class that inherits from the DatabasePlugin class. It implements the abstract methods of the DatabasePlugin class
    using Couchbase for database operations.

    Attributes:
        DATABASES (set): A set to store the databases.
        CREATED_COLLECTIONS (set): A set to store the created collections.
        UPDATED_COLLECTIONS (set): A set to store the updated collections.
        bucket (str): The bucket of the Couchbase server.
        scope (str): The scope of the Couchbase server.
        db (`couchbase.cluster.Cluster` instance): The connection to the Couchbase server.
    """

    DATABASES = set()
    CREATED_COLLECTIONS = set()
    UPDATED_COLLECTIONS = set()

    def __init__(self, uri, bucket, scope, username, password):
        """
        Initialize CouchBaseDatabase with the given uri, bucket, scope, username, and password for the Couchbase server.

        Args:
            uri (str): The uri of the Couchbase server.
            bucket (str): The bucket of the Couchbase server.
            scope (str): The scope of the Couchbase server.
            username (str): The username for the Couchbase server.
            password (str): The password for the Couchbase server.
        """
        auth = PasswordAuthenticator(username, password)

        self.bucket = bucket
        self.scope = scope

        self.db = Cluster(uri, ClusterOptions(auth)).bucket(bucket).scope(scope)

    async def _create_collection(self, collection):
        """
        Create a collection in the Couchbase server if it does not exist.

        Args:
            collection (str): The collection to be created.
        """
        self.db.query(
            f'CREATE COLLECTION `{self.bucket}`.{self.scope}.{collection.__collection__} ' f'if not exists',
            QueryOptions(scan_consistency=QueryScanConsistency.REQUEST_PLUS),
        ).execute()

    async def insert(self, document, *docs):
        """
        Insert one or more documents into the Couchbase server.

        Args:
            document (dict): The first document to be inserted.
            *docs (dict): Additional documents to be inserted.
        """
        for doc in docs:
            await self._create_collection(document.__metaclass__)

            values = '{' + ', '.join([f'"{k}" : ${k}' for k, v in doc.items()]) + '}'
            self.db.query(
                f'insert into `{self.bucket}`.{self.scope}.{document.__metaclass__.__collection__} '
                f'(key, value) '
                f'values '
                f"('{str(uuid.uuid4())}', {values}) "
                f'returning *',
                QueryOptions(
                    named_parameters={k: str(v) if isinstance(v, uuid.UUID) else v for k, v in doc.items()},
                    scan_consistency=QueryScanConsistency.REQUEST_PLUS,
                ),
            ).execute()

    async def read(self, document, query):
        """
        Read a document from the Couchbase server.

        Args:
            document (dict): The document to be read.
            query (dict): The query to be used for reading the document.

        Yields:
            dict: The read document.
        """
        await self._create_collection(document.__metaclass__)

        values = 'and '.join([f'`{k}` = ${k}' for k, v in query.items()])
        if len(values) > 0:
            result = self.db.query(
                f'SELECT * '
                f'FROM `{self.bucket}`.{self.scope}.{document.__metaclass__.__collection__} '
                f'WHERE {values}',
                QueryOptions(
                    named_parameters={k: str(v) if isinstance(v, uuid.UUID) else v for k, v in query.items()},
                    scan_consistency=QueryScanConsistency.REQUEST_PLUS,
                ),
            ).execute()
        else:
            result = self.db.query(
                f'SELECT * ' f'FROM `{self.bucket}`.{self.scope}.{document.__metaclass__.__collection__} ',
                QueryOptions(
                    named_parameters={k: str(v) if isinstance(v, uuid.UUID) else v for k, v in query.items()},
                    scan_consistency=QueryScanConsistency.REQUEST_PLUS,
                ),
            ).execute()
        for r in result:
            yield document(**r[document.__metaclass__.__collection__])

    async def update(self, document, query, update):
        """
        Update a document in the Couchbase server.

        Args:
            document (dict): The document to be updated.
            query (dict): The query to be used for updating the document.
            update (dict): The update to be applied to the document.
        """
        await self._create_collection(document.__metaclass__)

        query_values = 'and '.join([f'`{k}` = $query_{k}' for k, v in query.items()])
        update_values = ', '.join([f'`{k}` = $update_{k}' for k, v in update.items()])

        params = {}
        params.update({f'query_{k}': str(v) if isinstance(v, uuid.UUID) else v for k, v in query.items()})
        params.update({f'update_{k}': str(v) if isinstance(v, uuid.UUID) else v for k, v in update.items()})
        self.db.query(
            f'update `{self.bucket}`.{self.scope}.{document.__metaclass__.__collection__} '
            f'set {update_values} '
            f'WHERE {query_values}',
            QueryOptions(named_parameters=params, scan_consistency=QueryScanConsistency.REQUEST_PLUS),
        ).execute()

    async def delete(self, document, query):
        """
        Delete a document from the Couchbase server.

        Args:
            document (dict): The document to be deleted.
            query (dict): The query to be used for deleting the document.
        """
        await self._create_collection(document.__metaclass__)

        values = 'and '.join([f'`{k}` = ${k}' for k, v in query.items()])
        self.db.query(
            f'delete from `{self.bucket}`.{self.scope}.{document.__metaclass__.__collection__} ' f'WHERE {values}',
            QueryOptions(
                named_parameters={k: str(v) if isinstance(v, uuid.UUID) else v for k, v in query.items()},
                scan_consistency=QueryScanConsistency.REQUEST_PLUS,
            ),
        ).execute()

    async def start(self):
        """
        Start a database transaction.

        Raises:
            NotImplementedError: This method should be implemented in the child classes.
        """

    async def commit(self):
        """
        Commit a database transaction.

        Raises:
            NotImplementedError: This method should be implemented in the child classes.
        """

    async def abort(self):
        """
        Abort a database transaction.

        Raises:
            NotImplementedError: This method should be implemented in the child classes.
        """

    async def end(self):
        """
        End a database transaction.

        Raises:
            NotImplementedError: This method should be implemented in the child classes.
        """
