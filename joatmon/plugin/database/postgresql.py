import uuid
from datetime import datetime

import psycopg2

from joatmon import context
from joatmon.core.utility import get_converter
from joatmon.orm.constraint import UniqueConstraint
from joatmon.orm.document import Document
from joatmon.orm.meta import normalize_kwargs
from joatmon.plugin.database.core import DatabasePlugin


class PostgreSQLDatabase(DatabasePlugin):
    DATABASES = set()
    CREATED_COLLECTIONS = set()
    UPDATED_COLLECTIONS = set()

    def __init__(self, host, port, user, password, database):
        self.connection = psycopg2.connect(
            database=database, user=user,
            password=password, host=host, port=port  # , async_=True
        )
        self.connection.autocommit = True

    async def _check_collection(self, collection):
        cursor = self.connection.cursor()

        cursor.execute(f'select * from information_schema.tables where table_name = \'{collection.__collection__}\'')
        return len(list(cursor.fetchall())) > 0

    async def _create_collection(self, collection):
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
            fields.append(f'{field_name} {get_type(field.dtype)} {"" if field.nullable else "not null"} {"primary key" if field.primary else ""}')
        sql = f'create table {collection.__collection__} (\n' + ',\n'.join(fields) + '\n);'

        cursor = self.connection.cursor()

        cursor.execute(sql)

        index_names = set()
        for index_name, index in collection.constraints(collection).items():
            if ',' in index.field:
                index_fields = list(map(lambda x: x.strip(), index.field.split(',')))
            else:
                index_fields = [index.field]
            c = ", ".join(index_fields)
            if index_name in index_names:
                continue
            index_names.add(index_name)
            cursor.execute(f'create {"unique" if isinstance(index, UniqueConstraint) else ""} index {collection.__collection__}_{index_name} on {collection.__collection__} ({c})')

    async def _ensure_collection(self, collection):
        if not await self._check_collection(collection):
            await self._create_collection(collection)

    async def _get_collection(self, collection):
        ...

    async def drop_database(self):
        cursor = self.connection.cursor()
        cursor.execute(f'select table_name from  information_schema.tables where table_schema = \'public\'')
        for table, in cursor.fetchall():
            await self.drop_collection(table)

    async def drop_collection(self, collection):
        cursor = self.connection.cursor()
        cursor.execute(f'drop table {collection}')

    async def insert_raw(self, document):
        cursor = self.connection.cursor()

        if not isinstance(document, Document):
            raise ValueError(f'{type(document)} is not valid for saving')

        await self._ensure_collection(document.__metaclass__)

        def normalize(doc):
            dictionary = doc.validate()
            fields = doc.__metaclass__.fields(doc.__metaclass__)

            keys = []
            values = []
            for field_name, field in fields.items():
                keys.append(field_name)

                if dictionary[field_name] is None:
                    values.append('null')
                elif field.dtype in (uuid.UUID, str, datetime):
                    values.append(f'\'{str(dictionary[field_name])}\'')
                else:
                    values.append(str(dictionary[field_name]))

            return keys, values

        k, v = normalize(document)
        sql = f'insert into {document.__metaclass__.__collection__} ({", ".join(k)}) values ({", ".join(v)})'

        cursor.execute(sql)

        return document

    async def insert(self, *documents):
        for document in documents:
            user = context.get_value('user')
            document.creator_id = user.object_id if user is not None else uuid.UUID(int=0)
            document.created_at = datetime.utcnow()
            document.updater_id = user.object_id if user is not None else uuid.UUID(int=0)
            document.updated_at = datetime.utcnow()

            await self.insert_raw(document)

        return documents

    async def read(self, document, **kwargs):
        cursor = self.connection.cursor()

        await self._ensure_collection(document.__metaclass__)

        keys = list(document.__metaclass__.fields(document.__metaclass__).keys())

        sql = f'select {", ".join(keys)} from {document.__metaclass__.__collection__}'

        def normalize(doc, kwargs):
            fields = doc.__metaclass__.fields(doc.__metaclass__)

            keys = []
            values = []
            for k, v in kwargs.items():
                keys.append(k)

                field = fields[k]

                field_value = get_converter(field.dtype)(kwargs[k])

                if field_value is None:
                    values.append('null')
                elif field.dtype in (uuid.UUID, str, datetime):
                    values.append(f'\'{str(field_value)}\'')
                else:
                    values.append(str(field_value))

            return keys, values

        normalized = normalize_kwargs(document.__metaclass__, **kwargs)
        k, v = normalize(document, normalized)

        if len(kwargs) > 0:
            sql += f' where {" and ".join([f"{_k}={_v}" if _v != "null" else f"{_k} is {_v}" for _k, _v in zip(k, v)])}'

        cursor.execute(sql)

        # collection = await self._get_collection(document.__metaclass__.__collection__)
        # result = collection.find(
        #     normalize_kwargs(document.__metaclass__, **kwargs), {'_id': 0}, session=self.session
        # )

        for doc in cursor.fetchall():
            yield document(**dict(zip(keys, doc)))

    async def update_raw(self, document):
        cursor = self.connection.cursor()

        if not isinstance(document, Document):
            raise ValueError(f'{type(document)} is not valid for saving')

        def normalize(doc):
            dictionary = doc.validate()
            fields = doc.__metaclass__.fields(doc.__metaclass__)

            keys = []
            values = []
            for field_name, field in fields.items():
                keys.append(field_name)

                if dictionary[field_name] is None:
                    values.append('null')
                elif field.dtype in (uuid.UUID, str, datetime):
                    values.append(f'\'{str(dictionary[field_name])}\'')
                else:
                    values.append(str(dictionary[field_name]))

            return keys, values

        k, v = normalize(document)
        sql = f'update {document.__metaclass__.__collection__} set {", ".join(f"{_k}={_v}" for _k, _v in zip(k, v))} where object_id=\'{document.object_id}\''

        await self._ensure_collection(document.__metaclass__)
        cursor.execute(sql)

        return document

    async def update(self, *documents):
        for document in documents:
            user = context.get_value('user')
            document.updater_id = user.object_id if user is not None else uuid.UUID(int=0)
            document.updated_at = datetime.utcnow()

            await self.update_raw(document)

        return documents

    async def delete(self, *documents):
        for document in documents:
            if not isinstance(document, Document):
                raise ValueError(f'{type(document)} is not valid for saving')

            user = context.get_value('user')
            document.updater_id = user.object_id if user is not None else uuid.UUID(int=0)
            document.updated_at = datetime.utcnow()
            document.deleter_id = user.object_id if user is not None else uuid.UUID(int=0)
            document.deleted_at = datetime.utcnow()
            document.is_deleted = True

            await self.update_raw(document)

        return documents

    async def start(self):
        self.connection.autocommit = False

    async def commit(self):
        self.connection.commit()

    async def abort(self):
        self.connection.rollback()

    async def end(self):
        self.connection.close()
