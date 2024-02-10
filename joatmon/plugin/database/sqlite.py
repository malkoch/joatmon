import sqlite3
import uuid
from datetime import datetime

from joatmon.core.utility import get_converter
from joatmon.orm.constraint import UniqueConstraint
from joatmon.orm.meta import normalize_kwargs
from joatmon.orm.query import Dialects
from joatmon.plugin.database.core import DatabasePlugin


class SQLiteDatabase(DatabasePlugin):
    """
    PostgreSQLDatabase class that inherits from the DatabasePlugin class. It implements the abstract methods of the DatabasePlugin class
    using PostgreSQL for database operations.
    """

    # on del method
    def __init__(self, database):
        """
        Initialize PostgreSQLDatabase with the given host, port, user, password, and database for the PostgreSQL server.

        Args:
            database (str): The database of the PostgreSQL server.
        """
        self.connection = sqlite3.connect(f'{database}.sqlite3', isolation_level=None)

    async def _check_collection(self, collection):
        """
        Check if a collection exists in the PostgreSQL database.

        Args:
            collection (Meta): The collection to be checked.

        Returns:
            bool: True if the collection exists, False otherwise.
        """
        # for one time only need to check indexes, constraints, default values, table schema as well
        cursor = self.connection.cursor()

        cursor.execute(f"select * from sqlite_master where type='table' and name = '{collection.__collection__}'")
        return len(list(cursor.fetchall())) > 0

    async def _create_collection(self, collection):
        """
        Create a collection in the PostgreSQL database.

        Args:
            collection (Meta): The collection to be created.
        """

        def get_type(dtype: type):
            type_mapper = {
                datetime: 'timestamp without time zone',
                int: 'integer',
                float: 'real',
                str: 'varchar',
                bool: 'boolean',
                uuid.UUID: 'uuid',
            }

            return type_mapper.get(dtype, None)

        fields = []
        for field_name, field in collection.fields(collection).items():
            fields.append(
                f'{field_name} {get_type(field.dtype)} {"" if field.nullable else "not null"} {"primary key" if field.primary else ""}'
            )
        sql = f'create table {collection.__collection__} (\n' + ',\n'.join(fields) + '\n);'

        cursor = self.connection.cursor()

        cursor.execute(sql)

        index_names = set()
        for index_name, index in collection.constraints(collection).items():
            if ',' in index.field:
                index_fields = list(map(lambda x: x.strip(), index.field.split(',')))
            else:
                index_fields = [index.field]
            c = ', '.join(index_fields)
            if index_name in index_names:
                continue
            index_names.add(index_name)
            cursor.execute(
                f'create {"unique" if isinstance(index, UniqueConstraint) else ""} index {collection.__collection__}_{index_name} on {collection.__collection__} ({c})'
            )

    async def _create_view(self, collection):
        """
        Create a view for a collection in the PostgreSQL database.

        Args:
            collection (Meta): The collection for which the view is to be created.
        """
        cursor = self.connection.cursor()

        sql = f'drop VIEW if exists {collection.__collection__}'
        cursor.execute(sql)

        sql = f'CREATE OR REPLACE VIEW {collection.__collection__} AS {collection.query(collection).build(Dialects.POSTGRESQL)}'
        cursor.execute(sql)

    async def _ensure_collection(self, collection):
        """
        Ensure that a collection exists in the PostgreSQL database.

        Args:
            collection (Meta): The collection to be ensured.
        """
        if not await self._check_collection(collection):
            await self._create_collection(collection)

    async def create(self, document):
        """
        Create a new document in the PostgreSQL database.

        Args:
            document (Document): The document to be created.
        """
        await self._ensure_collection(document.__metaclass__)

    async def alter(self, document):
        """
        Alter an existing document in the PostgreSQL database.

        Args:
            document (dict): The document to be altered.

        Raises:
            NotImplementedError: This method should be implemented in the child classes.
        """

    async def drop(self, document):
        """
        Drop a collection from the PostgreSQL database.

        Args:
            document (Document): The document whose collection is to be dropped.
        """
        cursor = self.connection.cursor()
        sql = f'drop table if exists {document.__metaclass__.__collection__}'
        cursor.execute(sql)

    # @debug.timeit()
    async def insert(self, document, *docs):
        """
        Insert one or more documents into the PostgreSQL database.

        Args:
            document (Document): The first document to be inserted.
            *docs (dict): Additional documents to be inserted.
        """
        cursor = self.connection.cursor()

        for doc in docs:
            if not document.__metaclass__.structured:
                raise ValueError(f'you have to use structured document')

            await self._ensure_collection(document.__metaclass__)

            # @debug.timeit()
            async def normalize(d):
                dictionary = d.validate()
                fields = document.__metaclass__.fields(document.__metaclass__)

                keys = []
                values = []
                for field_name, field in fields.items():
                    keys.append(field_name)

                    if dictionary[field_name] is None:
                        values.append('null')
                    elif field.dtype in (uuid.UUID, str, datetime):
                        values.append(f"'{str(dictionary[field_name])}'")
                    else:
                        values.append(str(dictionary[field_name]))

                return keys, values, dictionary

            if isinstance(doc, dict):
                k, v, di = await normalize(document(**doc))
            elif isinstance(doc, document):
                k, v, di = await normalize(doc)
            else:
                raise ValueError(f'cannot convert object type {type(doc)} to {document}')
            sql = f'insert into {document.__metaclass__.__collection__} ({", ".join(k)}) values ({", ".join(v)})'

            cursor.execute(sql)

    async def read(self, document, query):
        """
        Read a document from the PostgreSQL database.

        Args:
            document (Document): The document to be read.
            query (dict): The query to be used for reading the document.

        Yields:
            dict: The read document.
        """
        cursor = self.connection.cursor()

        await self._ensure_collection(document.__metaclass__)

        keys = list(document.__metaclass__.fields(document.__metaclass__).keys())

        sql = f'select {", ".join(keys)} from {document.__metaclass__.__collection__}'

        def normalize(d, kwargs):
            fields = d.__metaclass__.fields(d.__metaclass__)

            keys = []
            values = []
            for k, v in kwargs.items():
                keys.append(k)

                field = fields[k]

                field_value = get_converter(field.dtype)(kwargs[k])

                if field_value is None:
                    values.append('null')
                elif field.dtype in (uuid.UUID, str, datetime):
                    values.append(f"'{str(field_value)}'")
                else:
                    values.append(str(field_value))

            return keys, values

        normalized = normalize_kwargs(document.__metaclass__, **query)
        k, v = normalize(document, normalized)

        if len(query) > 0:
            sql += f' where {" and ".join([f"{_k}={_v}" if _v != "null" else f"{_k} is {_v}" for _k, _v in zip(k, v)])}'

        cursor.execute(sql)

        # collection = await self._get_collection(document.__metaclass__.__collection__)
        # result = collection.find(
        #     normalize_kwargs(document.__metaclass__, **kwargs), {'_id': 0}, session=self.session
        # )

        for doc in cursor.fetchall():
            yield document(**dict(zip(keys, doc)))

    async def update(self, document, query, update):
        """
        Update a document in the PostgreSQL database.

        Args:
            document (Document): The document to be updated.
            query (dict): The query to be used for updating the document.
            update (dict): The update to be applied to the document.
        """
        cursor = self.connection.cursor()

        def normalize(d, kwargs):
            fields = d.__metaclass__.fields(d.__metaclass__)

            keys = []
            values = []
            for k, v in kwargs.items():
                keys.append(k)

                field = fields[k]

                field_value = get_converter(field.dtype)(kwargs[k])

                if field_value is None:
                    values.append('null')
                elif field.dtype in (uuid.UUID, str, datetime):
                    values.append(f"'{str(field_value)}'")
                else:
                    values.append(str(field_value))

            return keys, values

        k, v = normalize(document, update)

        sql = f'update {document.__metaclass__.__collection__} set {", ".join(f"{_k}={_v}" for _k, _v in zip(k, v))}'

        def normalize(d, kwargs):
            fields = d.__metaclass__.fields(d.__metaclass__)

            keys = []
            values = []
            for k, v in kwargs.items():
                keys.append(k)

                field = fields[k]

                field_value = get_converter(field.dtype)(kwargs[k])

                if field_value is None:
                    values.append('null')
                elif field.dtype in (uuid.UUID, str, datetime):
                    values.append(f"'{str(field_value)}'")
                else:
                    values.append(str(field_value))

            return keys, values

        normalized = normalize_kwargs(document.__metaclass__, **query)
        k, v = normalize(document, normalized)

        if len(query) > 0:
            sql += f' where {" and ".join([f"{_k}={_v}" if _v != "null" else f"{_k} is {_v}" for _k, _v in zip(k, v)])}'

        await self._ensure_collection(document.__metaclass__)
        cursor.execute(sql)

    async def delete(self, document, query):
        """
        Delete a document from the PostgreSQL database.

        Args:
            document (Document): The document to be deleted.
            query (dict): The query to be used for deleting the document.
        """
        cursor = self.connection.cursor()

        sql = f'delete from {document.__metaclass__.__collection__}'

        def normalize(d, kwargs):
            fields = d.__metaclass__.fields(d.__metaclass__)

            keys = []
            values = []
            for k, v in kwargs.items():
                keys.append(k)

                field = fields[k]

                field_value = get_converter(field.dtype)(kwargs[k])

                if field_value is None:
                    values.append('null')
                elif field.dtype in (uuid.UUID, str, datetime):
                    values.append(f"'{str(field_value)}'")
                else:
                    values.append(str(field_value))

            return keys, values

        normalized = normalize_kwargs(document.__metaclass__, **query)
        k, v = normalize(document, normalized)

        if len(query) > 0:
            sql += f' where {" and ".join([f"{_k}={_v}" if _v != "null" else f"{_k} is {_v}" for _k, _v in zip(k, v)])}'

        await self._ensure_collection(document.__metaclass__)
        cursor.execute(sql)

    async def view(self, document, query):
        """
        View a document in the PostgreSQL database.

        Args:
            document (Document): The document to be viewed.
            query (dict): The query to be used for viewing the document.

        Yields:
            dict: The viewed document.
        """
        await self._create_view(document.__metaclass__)

        keys = list(document.__metaclass__.fields(document.__metaclass__).keys())

        sql = f'select {", ".join(keys)} from {document.__metaclass__.__collection__}'

        def normalize(d, kwargs):
            fields = d.__metaclass__.fields(d.__metaclass__)

            keys = []
            values = []
            for k, v in kwargs.items():
                keys.append(k)

                field = fields[k]

                field_value = get_converter(field.dtype)(kwargs[k])

                if field_value is None:
                    values.append('null')
                elif field.dtype in (uuid.UUID, str, datetime):
                    values.append(f"'{str(field_value)}'")
                else:
                    values.append(str(field_value))

            return keys, values

        normalized = normalize_kwargs(document.__metaclass__, **query)
        k, v = normalize(document, normalized)

        if len(query) > 0:
            sql += f' where {" and ".join([f"{_k}={_v}" if _v != "null" else f"{_k} is {_v}" for _k, _v in zip(k, v)])}'

        cursor = self.connection.cursor()
        cursor.execute(sql)

        # keys = list(document.__metaclass__.fields(document.__metaclass__).keys())
        keys = [desc[0] for desc in cursor.description]
        for doc in cursor.fetchall():
            yield document(**dict(zip(keys, doc)))

    async def execute(self, document, query):
        """
        This is an abstract method that should be implemented in the child classes. It is used to
        execute a query on a document in the database.

        Args:
            document (dict): The document to be queried.
            query (dict): The query to be executed.

        Raises:
            NotImplementedError: This method should be implemented in the child classes.
        """

    async def count(self, query):
        """
        This is an abstract method that should be implemented in the child classes. It is used to
        count the number of documents that match a query in the database.

        Args:
            query (dict): The query to be counted.

        Raises:
            NotImplementedError: This method should be implemented in the child classes.
        """

    async def start(self):
        """
        Start a database transaction in the PostgreSQL database.

        This method sets the autocommit mode of the connection to False, which means that changes made to the database
        are not saved until you call the commit method.
        """
        self.connection.autocommit = False

    async def commit(self):
        """
        Commit a database transaction in the PostgreSQL database.

        This method saves the changes made to the database since the last call to the start method.
        """
        self.connection.commit()

    async def abort(self):
        """
        Abort a database transaction in the PostgreSQL database.

        This method discards the changes made to the database since the last call to the start method.
        """
        self.connection.rollback()

    async def end(self):
        """
        End a database transaction in the PostgreSQL database.

        This method closes the connection to the database. After calling this method, you cannot make any more
        queries to the database using this connection.
        """
        self.connection.close()
