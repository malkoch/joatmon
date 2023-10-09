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
    InsertOne,
    read_concern,
    UpdateMany,
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
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    DATABASES = set()
    CREATED_COLLECTIONS = set()
    UPDATED_COLLECTIONS = set()

    def __init__(self, uri, database):
        self.database_name = database
        self.client = pymongo.MongoClient(host=uri)
        self.database = self.client[database]

        self.session = None

    async def _check_collection(self, collection):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return collection.__collection__ in list(self.database.list_collection_names())

    async def _create_collection(self, collection):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """

        # even if collection is not structured, need to create indexes
        # even if collection is not structured, need to use default values. do not do full validate but do partial
        # need to write more converter. for numpy, torch etc.
        # need to write bulk writer
        # need to implement cache mechanism, so that when reading and writing it can be used as buffer to decrease database connection number
        # when using session, create a buffer and in the end apply all operations with bulk_write
        self.database.create_collection(collection.__collection__)

    async def _create_schema(self, collection):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
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
                list: ['array'],
                tuple: ['array'],
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
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
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
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if not await self._check_collection(collection):
            await self._create_collection(collection)
            await self._create_indexes(collection)

            if collection.structured and collection.force:
                await self._create_schema(collection)

    async def _get_collection(self, collection):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        codec_options = DEFAULT_CODEC_OPTIONS.with_options(uuid_representation=UuidRepresentation.STANDARD)
        if self.session is None:
            return self.database.get_collection(collection, codec_options=codec_options)
        else:
            return self.session.client[self.database_name].get_collection(collection, codec_options=codec_options)

    async def create(self, document):
        ...

    async def alter(self, document):
        ...

    async def drop(self, document):
        if self.session is None:
            self.database.drop_collection(document.__metaclass__.__collection__)
        else:
            self.session.client[self.database_name].drop_collection(document.__metaclass__.__collection__)

    async def insert(self, document, *docs):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
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

            to_insert.append(InsertOne(d))

        await self._ensure_collection(document.__metaclass__)
        collection = await self._get_collection(document.__metaclass__.__collection__)
        collection.bulk_write(to_insert, session=self.session)  # bulk write might be causing collection already in use error

    async def read(self, document, query):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
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
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if document.__metaclass__.structured and document.__metaclass__.force:
            query = normalize_kwargs(document.__metaclass__, **query)
            update = normalize_kwargs(document.__metaclass__, **update)

        await self._ensure_collection(document.__metaclass__)
        collection = await self._get_collection(document.__metaclass__.__collection__)

        to_update = [UpdateMany(dict(**query), {'$set': update}, upsert=True)]
        collection.bulk_write(to_update, session=self.session)  # bulk write might be causing collection already in use error

    async def delete(self, document, query):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if document.__metaclass__.structured and document.__metaclass__.force:
            query = normalize_kwargs(document.__metaclass__, **query)

        await self._ensure_collection(document.__metaclass__)
        collection = await self._get_collection(document.__metaclass__.__collection__)
        collection.bulk_write([DeleteMany(dict(**query))], session=self.session)  # bulk write might be causing collection already in use error

    async def view(self, document, query):
        ...

    async def execute(self, document, query):
        ...

    async def count(self, query):
        ...

    async def start(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.session = self.client.start_session()
        self.session.start_transaction(read_concern.ReadConcern('majority'), write_concern.WriteConcern('majority'))

    async def commit(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.session.commit_transaction()

    async def abort(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.session.abort_transaction()

    async def end(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.session.end_session()
