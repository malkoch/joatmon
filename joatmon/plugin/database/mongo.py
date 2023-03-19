import warnings

import pymongo
from bson.binary import UuidRepresentation
from bson.codec_options import DEFAULT_CODEC_OPTIONS
from pymongo import (
    read_concern,
    write_concern
)

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

    async def _get_collection(self, collection):
        codec_options = DEFAULT_CODEC_OPTIONS.with_options(uuid_representation=UuidRepresentation.STANDARD)
        if self.session is None:
            return self.database.get_collection(collection, codec_options=codec_options)
        else:
            return self.session.client[self.database_name].get_collection(collection, codec_options=codec_options)

    async def insert(self, document, *docs):
        for doc in docs:
            if document.__metaclass__.structured:
                warnings.warn(f'document validation will be ignored')

            collection = await self._get_collection(document.__metaclass__.__collection__)
            collection.insert_one(dict(**doc), session=self.session)

    async def read(self, document, query):
        collection = await self._get_collection(document.__metaclass__.__collection__)
        result = collection.find(dict(**query), {'_id': 0}, session=self.session)

        for doc in result:
            yield document(**doc)

    async def update(self, document, query, update):
        collection = await self._get_collection(document.__metaclass__.__collection__)
        collection.update_many(dict(**query), {'$set': dict(**update)}, session=self.session)

    async def delete(self, document, query):
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
