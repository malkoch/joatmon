import typing
import warnings
from collections import OrderedDict
from datetime import datetime
from uuid import UUID

import pymongo
from bson.binary import UuidRepresentation
from bson.codec_options import DEFAULT_CODEC_OPTIONS
from pymongo import (
    DeleteMany,
    read_concern,
    write_concern
)

from joatmon.orm.constraint import (
    PrimaryKeyConstraint,
    UniqueConstraint
)
from joatmon.orm.meta import normalize_kwargs
from joatmon.plugin.database.core import DatabasePlugin


class MongoDatabase(DatabasePlugin):
    """
    MongoDatabase class that inherits from the DatabasePlugin class. It implements the abstract methods of the DatabasePlugin class
    using MongoDB for database operations.

    Attributes:
        DATABASES (set): A set to store the databases.
        CREATED_COLLECTIONS (set): A set to store the created collections.
        UPDATED_COLLECTIONS (set): A set to store the updated collections.
        database_name (str): The name of the MongoDB database.
        client (`pymongo.MongoClient` instance): The connection to the MongoDB server.
        database (`pymongo.database.Database` instance): The MongoDB database instance.
        session (`pymongo.client_session.ClientSession` instance): The MongoDB client session instance.
    """

    DATABASES = set()
    CREATED_COLLECTIONS = set()
    UPDATED_COLLECTIONS = set()

    def __init__(self, uri, database):
        """
        Initialize MongoDatabase with the given uri and database for the MongoDB server.

        Args:
            uri (str): The uri of the MongoDB server.
            database (str): The name of the MongoDB database.
        """
        self.database_name = database
        self.client = pymongo.MongoClient(host=uri)
        self.database = self.client[database]

        self.session = None

    async def _check_collection(self, collection):
        """
        Check if a collection exists in the MongoDB database.

        Args:
            collection (Meta): The collection to be checked.

        Returns:
            bool: True if the collection exists, False otherwise.
        """
        return collection.__collection__ in list(self.database.list_collection_names())

    async def _create_collection(self, collection):
        """
        Create a collection in the MongoDB database.

        Args:
            collection (Meta): The collection to be created.
        """

        # even if collection is not structured, need to create indexes
        # even if collection is not structured, need to use default values. do not do full validate but do partial
        # need to write more converter.
        # need to write bulk writer
        # need to implement cache mechanism, so that when reading and writing it can be used as buffer to decrease database connection number
        # when using session, create a buffer and in the end apply all operations with bulk_write
        self.database.create_collection(collection.__collection__)

    async def _create_schema(self, collection):
        """
        Create a schema for a collection in the MongoDB database.

        Args:
            collection (Meta): The collection for which the schema is to be created.
        """

        def get_type(dtype: typing.Union[type, typing.List, typing.Tuple]):
            type_mapper = {
                datetime: ['date'],
                int: ['int'],
                float: ['double'],
                str: ['string'],
                bool: ['bool'],
                UUID: ['binData'],
                dict: ['object'],
                list: ['array2'],
                tuple: ['array2'],
                object: ['object'],
            }

            if isinstance(dtype, (tuple, list)):
                return sum(list(map(lambda x: type_mapper.get(x, ['object']), dtype)), [])
            else:
                return type_mapper.get(dtype, ['object'])

        vexpr = {
            '$jsonSchema': {
                'bsonType': 'object',
                'required': list(
                    map(lambda x: x[0], filter(lambda y: not y[1].nullable, collection.fields(collection).items()))
                ),
                'properties': {},
            }
        }
        for field_name, field in collection.fields(collection).items():
            # might want to rewrite this part again
            vexpr['$jsonSchema']['properties'][field_name] = {
                'bsonType': list(
                    set(get_type(field.dtype) if not field.nullable else get_type(field.dtype) + ['null'])
                ),
                'description': '',
            }

        cmd = OrderedDict(
            [  # indexes can be added here as well
                ('collMod', collection.__collection__),
                ('validator', vexpr),
                ('validationLevel', 'moderate'),
            ]
        )

        self.database.command(cmd)

    async def _create_indexes(self, collection):
        """
        Create indexes for a collection in the MongoDB database.

        Args:
            collection (Meta): The collection for which the indexes are to be created.
        """
        index_names = set()
        for index_name, index in collection.constraints(collection, lambda x: isinstance(x, (PrimaryKeyConstraint, UniqueConstraint))).items():
            if ',' in index.field:
                index_fields = list(map(lambda x: x.strip(), index.field.split(',')))
            else:
                index_fields = [index.field]
            c = [(f'{k}', 1) for k in index_fields]
            if index_name in index_names:
                continue
            index_names.add(index_name)
            try:
                self.database[collection.__collection__].create_index(c, unique=True, name=index_name)
            except Exception as ex:
                print(str(ex))
        for index_name, index in collection.indexes(collection).items():
            if ',' in index.field:
                index_fields = list(map(lambda x: x.strip(), index.field.split(',')))
            else:
                index_fields = [index.field]
            c = [(f'{k}', 1) for k in index_fields]
            if index_name in index_names:
                continue
            index_names.add(index_name)
            try:
                self.database[collection.__collection__].create_index(c, name=index_name)
            except Exception as ex:
                print(str(ex))

    async def _ensure_collection(self, collection):
        """
        Ensure that a collection exists in the MongoDB database.

        Args:
            collection (Meta): The collection to be ensured.
        """
        if not await self._check_collection(collection):
            await self._create_collection(collection)
            await self._create_indexes(collection)

            if collection.structured and collection.force:
                await self._create_schema(collection)

    async def _get_collection(self, collection):
        """
        Get a collection from the MongoDB database.

        Args:
            collection (Meta): The collection to be gotten.

        Returns:
            `pymongo.collection.Collection` instance: The gotten collection.
        """
        codec_options = DEFAULT_CODEC_OPTIONS.with_options(uuid_representation=UuidRepresentation.STANDARD)
        if self.session is None:
            return self.database.get_collection(collection, codec_options=codec_options)
        else:
            return self.session.client[self.database_name].get_collection(collection, codec_options=codec_options)

    async def create(self, document):
        """
        This is an abstract method that should be implemented in the child classes. It is used to
        create a new document in the database.

        Args:
            document (dict): The document to be created.

        Raises:
            NotImplementedError: This method should be implemented in the child classes.
        """

    async def alter(self, document):
        """
        This is an abstract method that should be implemented in the child classes. It is used to
        alter an existing document in the database.

        Args:
            document (dict): The document to be altered.

        Raises:
            NotImplementedError: This method should be implemented in the child classes.
        """

    async def drop(self, document):
        """
        Drop a collection from the MongoDB database.

        Args:
            document (Document): The document whose collection is to be dropped.
        """
        if self.session is None:
            self.database.drop_collection(document.__metaclass__.__collection__)
        else:
            self.session.client[self.database_name].drop_collection(document.__metaclass__.__collection__)

    async def insert(self, document, *docs):
        """
        Insert one or more documents into the MongoDB database.

        Args:
            document (Document): The first document to be inserted.
            *docs (Union[dict, Document]): Additional documents to be inserted.
        """

        to_insert = []
        for doc in docs:
            d = dict(**doc)

            if document.__metaclass__.structured and document.__metaclass__.force:
                if isinstance(doc, document):
                    d = doc.validate()
                else:
                    d = document(**d).validate()
            elif document.__metaclass__.structured:
                warnings.warn(f'document validation will be ignored')

            to_insert.append(d)

        await self._ensure_collection(document.__metaclass__)
        collection = await self._get_collection(document.__metaclass__.__collection__)
        collection.insert_many(to_insert, session=self.session, ordered=False)  # bulk write might be causing collection already in use error

    async def read(self, document, query):
        """
        Read a document from the MongoDB database.

        Args:
            document (Document): The document to be read.
            query (dict): The query to be used for reading the document.

        Yields:
            dict: The read document.
        """
        if document.__metaclass__.structured and document.__metaclass__.force:
            query = normalize_kwargs(document.__metaclass__, **query)

        await self._ensure_collection(document.__metaclass__)
        collection = await self._get_collection(document.__metaclass__.__collection__)
        result = collection.find(dict(**query), {'_id': 0}, session=self.session)

        for doc in result:
            yield document(**doc)

    async def update(self, document, query, update):
        """
        Update a document in the MongoDB database.

        Args:
            document (Document): The document to be updated.
            query (dict): The query to be used for updating the document.
            update (dict): The update to be applied to the document.
        """
        if document.__metaclass__.structured and document.__metaclass__.force:
            query = normalize_kwargs(document.__metaclass__, **query)
            update = normalize_kwargs(document.__metaclass__, **update)

        await self._ensure_collection(document.__metaclass__)
        collection = await self._get_collection(document.__metaclass__.__collection__)

        collection.update_many(dict(**query), {'$set': update}, upsert=True, session=self.session)  # bulk write might be causing collection already in use error

    async def delete(self, document, query):
        """
        Delete a document from the MongoDB database.

        Args:
            document (Document): The document to be deleted.
            query (dict): The query to be used for deleting the document.
        """
        if document.__metaclass__.structured and document.__metaclass__.force:
            query = normalize_kwargs(document.__metaclass__, **query)

        await self._ensure_collection(document.__metaclass__)
        collection = await self._get_collection(document.__metaclass__.__collection__)
        collection.delete_many(dict(**query), session=self.session)  # bulk write might be causing collection already in use error

    async def view(self, document, query):
        """
        This is an abstract method that should be implemented in the child classes. It is used to
        view a document in the database.

        Args:
            document (dict): The document to be viewed.
            query (dict): The query to be used for viewing the document.

        Raises:
            NotImplementedError: This method should be implemented in the child classes.
        """

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
        Start a database transaction in the MongoDB database.
        """
        self.session = self.client.start_session()
        self.session.start_transaction(read_concern.ReadConcern('majority'), write_concern.WriteConcern('majority'))

    async def commit(self):
        """
        Commit a database transaction in the MongoDB database.
        """
        self.session.commit_transaction()

    async def abort(self):
        """
        Abort a database transaction in the MongoDB database.
        """
        self.session.abort_transaction()

    async def end(self):
        """
        End a database transaction in the MongoDB database.
        """
        self.session.end_session()
