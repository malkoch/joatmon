import json
import typing
from collections import OrderedDict
from datetime import datetime
from uuid import UUID

import pymongo
from pymongo import (
    read_concern,
    write_concern
)

from joatmon.context import current
from joatmon.orm.constraint import UniqueConstraint
from joatmon.orm.document import Document
from joatmon.orm.meta import Meta
from joatmon.orm.query import QueryBuilder
from joatmon.orm.utility import (
    empty_object_id,
    normalize_kwargs
)
from joatmon.plugin.database.core import Database


class MongoDatabase(Database):
    def __init__(self, alias: str, host: str, port: int, database: str, username: str, password: str, auth_database: str):
        super(MongoDatabase, self).__init__(alias)

        connection = f'mongodb://{username}:{password}@{host}:{port}/{auth_database}'

        self._database_name = database
        self._client = pymongo.MongoClient(connection, replicaset='rs0')
        self._database = self._client[self._database_name]

        self.session = None

    def start(self):
        self.session = self._client.start_session()
        self.session.start_transaction(read_concern.ReadConcern('snapshot'), write_concern.WriteConcern('majority'))

    def commit(self):
        self.session.commit_transaction()

    def abort(self):
        self.session.abort_transaction()

    def end(self):
        self.session.end_session()

    def drop(self):
        for collection_name in self.session.client[self._database_name].list_collection_names():
            self.session.client[self._database_name][collection_name].drop()

    def initialize(self):
        def get_type(kind: typing.Union[type, typing.List, typing.Tuple]):
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

            if isinstance(kind, (tuple, list)):
                return sum(list(map(lambda x: type_mapper.get(x, ['object']), kind)), [])
            else:
                return type_mapper.get(kind, ['object'])

        for document_name, document in Meta.subclasses(Meta).items():  # need to check collection info from mongo, if field is just added need to put default value of fallback
            if document_name not in self.session.client[self._database_name].list_collection_names():
                self.session.client[self._database_name].create_collection(document_name)

            vexpr = {
                '$jsonSchema': {
                    'bsonType': 'object',
                    'required': list(map(lambda x: x[0], filter(lambda y: not y[1].nullable, document.fields(document).items()))),
                    'properties': {}
                }
            }
            for field_name, field in document.fields(document).items():
                # might want to rewrite this part again
                vexpr['$jsonSchema']['properties'][field_name] = {
                    'bsonType': list(set(get_type(field.kind) if not field.nullable else get_type(field.kind) + ['null'])),
                    'description': ''
                }

            cmd = OrderedDict([  # indexes can be added here as well
                ('collMod', document.__collection__),
                ('validator', vexpr),
                ('validationLevel', 'moderate')
            ])

            self.session.client[self._database_name].command(cmd)

            index_names = set()
            for constraint_name, constraint in document.constraints(document, predicate=lambda x: isinstance(x, UniqueConstraint)).items():
                if ',' in constraint.field:
                    constraint_fields = list(map(lambda x: x.strip(), constraint.field.split(',')))
                else:
                    constraint_fields = [constraint.field]
                c = [(f'{k}', 1) for k in constraint_fields]
                if constraint_name in index_names:
                    continue
                index_names.add(constraint_name)
                try:
                    self.session.client[self._database_name][document_name].create_index(c, unique=True, name=constraint_name)
                except Exception as ex:
                    print(str(ex))

            for index_name, index in document.indexes(document).items():
                if ',' in index.field:
                    index_fields = list(map(lambda x: x.strip(), index.field.split(',')))
                else:
                    index_fields = [index.field]
                c = [(f'{k}', 1) for k in index_fields]
                if index_name in index_names:
                    continue
                index_names.add(index_name)
                try:
                    self.session.client[self._database_name][document_name].create_index(c, name=index_name)
                except Exception as ex:
                    print(str(ex))

    def save(self, *documents):
        current_user_id = (current['user'].object_id if current['user'] is not None else None) or empty_object_id()

        for document in documents:
            if not isinstance(document, Document):
                raise ValueError(f'{type(document)} is not valid for saving')

            document.creator_id = current_user_id
            document.created_at = datetime.utcnow()
            document.updater_id = current_user_id
            document.updated_at = datetime.utcnow()

            dictionary = document.validate()

            self.session.client[self._database_name][document.__metaclass__.__collection__].insert_one(dictionary, session=self.session)

        return documents

    def read(self, document_type: type, **kwargs):
        result = self.session.client[self._database_name][document_type.__metaclass__.__collection__].find(
            normalize_kwargs(document_type.__metaclass__, **kwargs), {'_id': 0}, session=self.session
        )

        for document in result:
            yield document_type(**document)

    def update(self, *documents):
        current_user_id = (current['user'].object_id if current['user'] is not None else None) or empty_object_id()

        for document in documents:
            if not isinstance(document, Document):
                raise ValueError(f'{type(document)} is not valid for saving')

            document.updater_id = current_user_id
            document.updated_at = datetime.utcnow()

            dictionary = document.validate()

            query = {'object_id': document.object_id}
            update = {'$set': dictionary}

            self.session.client[self._database_name][document.__metaclass__.__collection__].update_one(query, update, session=self.session)

        return documents

    def delete(self, *documents):
        current_user_id = (current['user'].object_id if current['user'] is not None else None) or empty_object_id()

        for document in documents:
            if not isinstance(document, Document):
                raise ValueError(f'{type(document)} is not valid for saving')

            document.updater_id = current_user_id
            document.updated_at = datetime.utcnow()
            document.deleter_id = current_user_id
            document.deleted_at = datetime.utcnow()
            document.is_deleted = True
            self.update(document)
        return documents

    def execute(self, query: QueryBuilder):
        for document in self.session.client[self._database_name][query.collection].aggregate(query.build().aggregation):
            yield document
