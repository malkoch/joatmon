import uuid
from datetime import datetime

import psycopg2

from joatmon.core.utility import get_converter
from joatmon.orm.constraint import UniqueConstraint
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
        # for one time only need to check indexes, constraints, default values, table schema as well
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

    async def insert(self, document, *docs):
        cursor = self.connection.cursor()

        for doc in docs:
            if not document.__metaclass__.structured:
                raise ValueError(f'you have to use structured document')

            await self._ensure_collection(document.__metaclass__)

            def normalize(d):
                dictionary = d.validate()
                fields = document.__metaclass__.fields(document.__metaclass__)

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

                return keys, values, dictionary

            k, v, di = normalize(document(**doc))
            sql = f'insert into {document.__metaclass__.__collection__} ({", ".join(k)}) values ({", ".join(v)})'

            cursor.execute(sql)

            yield document(**di)

    async def read(self, document, query):
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
                    values.append(f'\'{str(field_value)}\'')
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
                    values.append(f'\'{str(field_value)}\'')
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
                    values.append(f'\'{str(field_value)}\'')
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
                    values.append(f'\'{str(field_value)}\'')
                else:
                    values.append(str(field_value))

            return keys, values

        normalized = normalize_kwargs(document.__metaclass__, **query)
        k, v = normalize(document, normalized)

        if len(query) > 0:
            sql += f' where {" and ".join([f"{_k}={_v}" if _v != "null" else f"{_k} is {_v}" for _k, _v in zip(k, v)])}'

        await self._ensure_collection(document.__metaclass__)
        cursor.execute(sql)

    async def start(self):
        self.connection.autocommit = False

    async def commit(self):
        self.connection.commit()

    async def abort(self):
        self.connection.rollback()

    async def end(self):
        self.connection.close()
