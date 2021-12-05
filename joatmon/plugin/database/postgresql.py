from datetime import datetime

import psycopg2
import psycopg2.extras

from joatmon.context import current
from joatmon.database.constraint import UniqueConstraint
from joatmon.database.document import Document
from joatmon.database.utility import (
    empty_object_id,
    normalize_kwargs
)
from joatmon.plugin.database.core import Database


class PostgreSQLDatabase(Database):
    def __init__(self, alias: str, host: str, port: int, database: str, username: str, password: str):
        super(PostgreSQLDatabase, self).__init__(alias)
        psycopg2.extras.register_uuid()

        connection = f"host={host} port={port} dbname={database} user={username} password={password}"
        # self._connection = psycopg2.connect(connection, async_=True)
        self._connection = psycopg2.connect(connection)
        self._connection.autocommit = False

        self._connection.cursor()

        self.session = None

    def start(self):
        session = self._connection.cursor()
        return session

    def commit(self):
        self._connection.commit()

    def abort(self):
        self._connection.rollback()

    def end(self):
        self.session.close()
        self._connection.close()

    def drop(self):
        self.session.execute("SELECT * FROM pg_catalog.pg_tables where tableowner = 'joatmon' and schemaname = 'public';")
        results = self.session.fetchall()
        for result in results:
            table_name = result[1]
            self.session.execute(f'drop table {table_name}')

    def initialize(self):
        def get_type(name):
            type_mapper = {
                'datetime': 'timestamp without time zone',
                'int': 'integer',
                'integer': 'integer',
                'float': 'double',
                'str': 'character varying',
                'string': 'character varying',
                'bool': 'boolean',
                'boolean': 'boolean',
                'uuid': 'uuid',
                'dict': 'object',
                'dictionary': 'object',
                'list': 'array',
                'tuple': 'array',
                'object': 'object',
                'enumerable': ['array', 'object', 'string']
            }
            return type_mapper.get(name.lower(), 'unknown')

        for collection_name, document in Document.subclasses().items():
            create_sql = f"""
                create table public.{collection_name} (
                    {",".join([f'{field.name} {get_type(field.kind)} {"" if field.nullable else "not null"}' for field in document.fields().values()])}
                );
                
                alter table public.{collection_name} owner to joatmon;
            """
            self.session.execute(create_sql)

            index_names = set()
            for constraint in document.constraints():
                if isinstance(constraint, UniqueConstraint):
                    if ',' in constraint.field:
                        constraint_fields = list(map(lambda x: x.strip(), constraint.field.split(',')))
                    else:
                        constraint_fields = [constraint.field]
                    c = [(f'{k}', 1) for k in constraint_fields]
                    index_name = '_'.join(k[0] for k in c)
                    if index_name in index_names:
                        continue
                    index_names.add(index_name)

                    index_sql = f"""
                        create unique index {collection_name}_unique_{index_name}
                            on public.{collection_name} using btree
                            ({','.join([f'{field_name} asc nulls last' for field_name in constraint_fields])})
                            tablespace pg_default
                    """
                    self.session.execute(index_sql)

            for index in document.indexes():
                if ',' in index.field:
                    index_fields = list(map(lambda x: x.strip(), index.field.split(',')))
                else:
                    index_fields = [index.field]
                c = [(f'{k}', 1) for k in index_fields]
                index_name = '_'.join(k[0] for k in c)
                if index_name in index_names:
                    continue
                index_names.add(index_name)

                index_sql = f"""
                    create index {collection_name}_index_{index_name}
                        on public.{collection_name} using btree
                        ({','.join([f'{field_name} asc nulls last' for field_name in index_fields])})
                        tablespace pg_default
                """
                self.session.execute(index_sql)

    def save(self, *documents):
        current_user_id = (current['user'].object_id if current['user'] is not None else None) or empty_object_id()

        for document in documents:
            if not isinstance(document, Document):
                raise ValueError(f'{type(document)} is not valid for saving')

            document.creator_id = current_user_id
            document.created_at = datetime.utcnow()
            document.updater_id = current_user_id
            document.updated_at = datetime.utcnow()

            document.validate()

            dictionary = document.write_dict
            keys = dictionary.keys()

            insert_sql = f"""
                insert into {document.CollectionName}({','.join(keys)}) values ({','.join(['%s' for _ in keys])})
            """
            self.session.execute(insert_sql, [dictionary[key] for key in keys])

        return documents

    def read(self, document_type, **kwargs):
        normalized_kwargs = normalize_kwargs(document_type, **kwargs)
        keys = normalized_kwargs.keys()
        read_sql = f"""
            select * from {document_type.CollectionName} {'' if len(kwargs) == 0 else f"where {' '.join([f'{key}=%s' for key in keys])}"}
        """
        self.session.execute(read_sql, [normalized_kwargs[key] for key in keys])
        result = self.session.fetchall()
        column_names = [desc[0] for desc in self.session.description]

        for document in result:
            yield document_type.from_dict(dict(zip(column_names, document)))

    def update(self, *documents):
        current_user_id = (current['user'].object_id if current['user'] is not None else None) or empty_object_id()

        for document in documents:
            if not isinstance(document, Document):
                raise ValueError(f'{type(document)} is not valid for saving')

            document.updater_id = current_user_id
            document.updated_at = datetime.utcnow()

            document.validate()

            dictionary = document.write_dict
            keys = dictionary.keys()

            update_sql = f"""
                update {document.CollectionName} set {','.join([f'{key}=%s' for key in keys])} where object_id=%s
            """
            self.session.execute(update_sql, [dictionary[key] for key in keys] + [document.object_id])

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
