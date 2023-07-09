import typing
import warnings
from collections import OrderedDict
from datetime import datetime
from uuid import UUID

import pymongo
from bson.binary import UuidRepresentation
from bson.codec_options import DEFAULT_CODEC_OPTIONS
from pymongo import (
    read_concern,
    write_concern
)

from joatmon.orm.constraint import UniqueConstraint
from joatmon.plugin.database.core import DatabasePlugin


class MongoDatabase(DatabasePlugin):
    DATABASES = set()
    CREATED_COLLECTIONS = set()
    UPDATED_COLLECTIONS = set()

    def __init__(self, uri, database):
        self.database_name = database
        self.client = pymongo.MongoClient(host=uri)
        self.database = self.client[database]

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

    async def insert(self, document, *docs):
        for doc in docs:
            d = dict(**doc)

            if document.__metaclass__.structured and document.__metaclass__.force:
                await self._ensure_collection(document.__metaclass__)
                d = document(**doc).validate()
            elif document.__metaclass__.structured:
                warnings.warn(f'document validation will be ignored')

            collection = await self._get_collection(document.__metaclass__.__collection__)
            collection.insert_one(d, session=self.session)

    async def read(self, document, query):
        if document.__metaclass__.structured and document.__metaclass__.force:
            await self._ensure_collection(document.__metaclass__)
        elif document.__metaclass__.structured:
            warnings.warn(f'document validation will be ignored')

        collection = await self._get_collection(document.__metaclass__.__collection__)
        result = collection.find(dict(**query), {'_id': 0}, session=self.session)

        for doc in result:
            yield document(**doc)

    async def update(self, document, query, update):
        if document.__metaclass__.structured and document.__metaclass__.force:
            await self._ensure_collection(document.__metaclass__)
        elif document.__metaclass__.structured:
            warnings.warn(f'document validation will be ignored')

        collection = await self._get_collection(document.__metaclass__.__collection__)
        collection.update_many(dict(**query), {'$set': dict(**update)}, session=self.session)

    async def delete(self, document, query):
        if document.__metaclass__.structured and document.__metaclass__.force:
            await self._ensure_collection(document.__metaclass__)
        elif document.__metaclass__.structured:
            warnings.warn(f'document validation will be ignored')

        collection = await self._get_collection(document.__metaclass__.__collection__)
        collection.delete_many(dict(**query), session=self.session)

    async def start(self):
        self.session = self.client.start_session()
        self.session.start_transaction(read_concern.ReadConcern('majority'), write_concern.WriteConcern('majority'))

    async def commit(self):
        self.session.commit_transaction()

    async def abort(self):
        self.session.abort_transaction()

    async def end(self):
        self.session.end_session()
