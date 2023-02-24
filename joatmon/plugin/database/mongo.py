import typing
import uuid
from collections import OrderedDict
from datetime import datetime
from uuid import UUID

import pymongo
from bson.binary import UuidRepresentation
from bson.codec_options import DEFAULT_CODEC_OPTIONS
from pymongo import read_concern, write_concern

from joatmon import context
from joatmon.orm.constraint import UniqueConstraint
from joatmon.orm.document import Document
from joatmon.orm.meta import normalize_kwargs
from joatmon.plugin.database.core import DatabasePlugin


class MongoDatabase(DatabasePlugin):
    DATABASES = set()
    CREATED_COLLECTIONS = set()
    UPDATED_COLLECTIONS = set()

    def __init__(self, uri, database, user_plugin):
        self.database_name = database
        self.client = pymongo.MongoClient(host=uri)
        self.database = self.client[database]
        self.user_plugin = user_plugin

        self.session = None

    async def _check_collection(self, collection):
        return collection.__collection__ in list(self.database.list_collection_names())

    async def _create_collection(self, collection):
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
                object: ['object']
            }

            if isinstance(dtype, (tuple, list)):
                return sum(list(map(lambda x: type_mapper.get(x, ['object']), dtype)), [])
            else:
                return type_mapper.get(dtype, ['object'])

        self.database.create_collection(
            collection.__collection__
        )

        vexpr = {
            '$jsonSchema': {
                'bsonType': 'object',
                'required': list(
                    map(lambda x: x[0], filter(lambda y: not y[1].nullable, collection.fields(collection).items()))
                ),
                'properties': {}
            }
        }
        for field_name, field in collection.fields(collection).items():
            # might want to rewrite this part again
            vexpr['$jsonSchema']['properties'][field_name] = {
                'bsonType': list(
                    set(get_type(field.dtype) if not field.nullable else get_type(field.dtype) + ['null'])
                ),
                'description': ''
            }

        cmd = OrderedDict(
            [  # indexes can be added here as well
                ('collMod', collection.__collection__),
                ('validator', vexpr),
                ('validationLevel', 'moderate')
            ]
        )

        self.database.command(cmd)

        index_names = set()
        for index_name, index in collection.constraints(collection).items():
            if ',' in index.field:
                index_fields = list(map(lambda x: x.strip(), index.field.split(',')))
            else:
                index_fields = [index.field]
            c = [(f'{k}', 1) for k in index_fields]
            if index_name in index_names:
                continue
            index_names.add(index_name)
            try:
                self.database[collection.__collection__].create_index(c, unique=isinstance(index, UniqueConstraint), name=index_name)
            except Exception as ex:
                print(str(ex))

    async def _ensure_collection(self, collection):
        if not await self._check_collection(collection):
            await self._create_collection(collection)

    async def _get_collection(self, collection):
        codec_options = DEFAULT_CODEC_OPTIONS.with_options(uuid_representation=UuidRepresentation.STANDARD)
        if self.session is None:
            return self.database.get_collection(collection, codec_options=codec_options)
        else:
            return self.session.client[self.database_name].get_collection(collection, codec_options=codec_options)

    async def drop_database(self):
        for collection_name in self.database.list_collection_names():
            self.database.drop_collection(collection_name)

    async def drop_collection(self, collection):
        self.database.drop_collection(collection.__collection__)

    async def insert_raw(self, document):
        if not isinstance(document, Document):
            raise ValueError(f'{type(document)} is not valid for saving')

        await self._ensure_collection(document.__metaclass__)

        dictionary = document.validate()

        collection = await self._get_collection(document.__metaclass__.__collection__)
        collection.insert_one(dictionary, session=self.session)

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
        result = collection.find(
            normalize_kwargs(document.__metaclass__, **kwargs), {'_id': 0}, session=self.session
        )

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
        collection.update_one(query, update, session=self.session)

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
        self.session = self.client.start_session()
        self.session.start_transaction(read_concern.ReadConcern('majority'), write_concern.WriteConcern('majority'))

    async def commit(self):
        self.session.commit_transaction()

    async def abort(self):
        self.session.abort_transaction()

    async def end(self):
        self.session.end_session()
