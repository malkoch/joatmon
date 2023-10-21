import uuid
from datetime import datetime

import psycopg2

from joatmon.orm.constraint import UniqueConstraint
from joatmon.orm.meta import normalize_kwargs
from joatmon.orm.query import Dialects
from joatmon.plugin.database.core import DatabasePlugin
from joatmon.core.utility import get_converter


class PostgreSQLDatabase(DatabasePlugin):
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

    # on del method
    def __init__(self, host, port, user, password, database):
        self.connection = psycopg2.connect(
            database=database, user=user, password=password, host=host, port=port  # , async_=True
        )
        self.connection.autocommit = True

    async def _check_collection(self, collection):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        # for one time only need to check indexes, constraints, default values, table schema as well
        cursor = self.connection.cursor()

        cursor.execute(f"select * from information_schema.tables where table_name = '{collection.__collection__}'")
        return len(list(cursor.fetchall())) > 0

    async def _create_collection(self, collection):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
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
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        cursor = self.connection.cursor()

        sql = f'drop VIEW if exists {collection.__collection__}'
        cursor.execute(sql)

        sql = f'CREATE OR REPLACE VIEW {collection.__collection__} AS {collection.query(collection).build(Dialects.POSTGRESQL)}'
        cursor.execute(sql)

    async def _ensure_collection(self, collection):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if not await self._check_collection(collection):
            await self._create_collection(collection)

    async def create(self, document):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        await self._ensure_collection(document.__metaclass__)

    async def alter(self, document):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...

    async def drop(self, document):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        cursor = self.connection.cursor()
        sql = f'drop table if exists {document.__metaclass__.__collection__} cascade'
        cursor.execute(sql)

    # @debug.timeit()
    async def insert(self, document, *docs):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
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
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
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
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
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
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
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
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
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
        self.connection.autocommit = False

    async def commit(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.connection.commit()

    async def abort(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.connection.rollback()

    async def end(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.connection.close()
