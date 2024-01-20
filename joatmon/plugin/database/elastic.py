import warnings

from elasticsearch import Elasticsearch

from joatmon.plugin.database.core import DatabasePlugin


class ElasticDatabase(DatabasePlugin):
    """
    ElasticDatabase class that inherits from the DatabasePlugin class. It implements the abstract methods of the DatabasePlugin class
    using Elasticsearch for database operations.

    Attributes:
        DATABASES (set): A set to store the databases.
        CREATED_COLLECTIONS (set): A set to store the created collections.
        UPDATED_COLLECTIONS (set): A set to store the updated collections.
        client (`elasticsearch.Elasticsearch` instance): The connection to the Elasticsearch server.
    """

    DATABASES = set()
    CREATED_COLLECTIONS = set()
    UPDATED_COLLECTIONS = set()

    def __init__(self, uri):
        """
        Initialize ElasticDatabase with the given uri for the Elasticsearch server.

        Args:
            uri (str): The uri of the Elasticsearch server.
        """
        self.client = Elasticsearch(uri)

    async def insert(self, document, *docs):
        """
        Insert one or more documents into the Elasticsearch server.

        Args:
            document (dict): The first document to be inserted.
            *docs (dict): Additional documents to be inserted.
        """
        for doc in docs:
            if document.__metaclass__.structured:
                warnings.warn(f'document validation will be ignored')

            self.client.index(index=document.__metaclass__.__collection__, document=dict(**doc))

    async def read(self, document, query):
        """
        Read a document from the Elasticsearch server.

        Args:
            document (dict): The document to be read.
            query (dict): The query to be used for reading the document.

        Raises:
            NotImplementedError: This method should be implemented in the child classes.
        """

    async def update(self, document, query, update):
        """
        Update a document in the Elasticsearch server.

        Args:
            document (dict): The document to be updated.
            query (dict): The query to be used for updating the document.
            update (dict): The update to be applied to the document.

        Raises:
            NotImplementedError: This method should be implemented in the child classes.
        """

    async def delete(self, document, query):
        """
        Delete a document from the Elasticsearch server.

        Args:
            document (dict): The document to be deleted.
            query (dict): The query to be used for deleting the document.

        Raises:
            NotImplementedError: This method should be implemented in the child classes.
        """

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
