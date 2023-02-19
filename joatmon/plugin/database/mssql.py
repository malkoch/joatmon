from datetime import datetime

import pypyodbc

from joatmon.context import current
from joatmon.orm.constraint import UniqueConstraint
from joatmon.orm.document import Document
from joatmon.orm.utility import (
    empty_object_id,
    normalize_kwargs
)
from joatmon.plugin.database.core import Database


class MSSQLDatabase(Database):
    def __init__(self, alias: str, host: str, port: int, database: str, username: str, password: str):
        super(MSSQLDatabase, self).__init__(alias)

        connection = f"DRIVER={{ODBC Driver 13 for SQL Server}};SERVER={host};DATABASE={database};UID={username};PWD={password}"
        self._connection = pypyodbc.connect(connection)
        self._connection.autocommit = False

        self._connection.cursor()

        self.session = None

    @staticmethod
    def _field_create(field):
        type_mapper = {
            'datetime': 'datetime',
            'int': 'int',
            'integer': 'int',
            'float': 'double',
            'str': 'nvarchar(MAX)',
            'string': 'nvarchar(MAX)',
            'bool': 'bit',
            'boolean': 'bit',
            'uuid': 'uniqueidentifier',
            'dict': 'object',
            'dictionary': 'object',
            'list': 'array',
            'tuple': 'array',
            'object': 'object',
            'enumerable': ['array', 'object', 'string']
        }
        return f'{field.name} {type_mapper.get(field.kind.lower(), "unknown")} {"" if field.nullable else "not null"}'

    @staticmethod
    def _field_insert(field, value):
        type_mapper = {
            'datetime': lambda x: x.strftime('%Y-%m-%d %H:%M:%S'),
            'uuid': lambda x: str(x)
        }
        if value is None:
            return None
        return f'{type_mapper.get(field.kind.lower(), lambda x: x)(value)}'

    @staticmethod
    def _field_select(field):
        type_mapper = {
            'uuid': lambda x: f"cast({x.name} as varchar(36)) as {x.name}"
        }
        return f'{type_mapper.get(field.kind.lower(), lambda x: x.name)(field)}'

    @staticmethod
    def _field_update(field, value):
        type_mapper = {
            'datetime': lambda x: x.strftime('%Y-%m-%d %H:%M:%S'),
            'uuid': lambda x: str(x)
        }
        if value is None:
            return None
        return f'{type_mapper.get(field.kind.lower(), lambda x: x)(value)}'

    @staticmethod
    def _field_where(field, value):
        type_mapper = {
            'datetime': lambda x: x.strftime('%Y-%m-%d %H:%M:%S'),
            'uuid': lambda x: str(x)
        }
        if value is None:
            return None
        return f'{type_mapper.get(field.kind.lower(), lambda x: x)(value)}'

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
        self.session.execute("SELECT * FROM sys.tables;")
        results = self.session.fetchall()
        for result in results:
            table_name = result[0]
            self.session.execute(f'drop table {table_name}')

    def initialize(self):
        for collection_name, document in Document.subclasses().items():
            create_sql = f"""
                create table {collection_name} (
                    {",".join([MSSQLDatabase._field_create(field) for field in document.fields().values()])}
                );
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
                        on {collection_name}
                        ({','.join([f'{field_name} asc' for field_name in constraint_fields])})
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
                    on {collection_name}
                    ({','.join([f'{field_name} asc' for field_name in index_fields])})
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
                insert into 
                {document.CollectionName} 
                ({', '.join(keys)}) 
                values 
                ({', '.join(['?' for _ in keys])})
            """
            self.session.execute(insert_sql, [MSSQLDatabase._field_insert({f.name: f for f in document.fields().values()}[key], dictionary[key]) for key in keys])

        return documents

    def read(self, document_type, **kwargs):
        normalized_kwargs = normalize_kwargs(document_type, **kwargs)
        keys = normalized_kwargs.keys()
        read_sql = f"""
            select 
            {','.join([MSSQLDatabase._field_select(field) for field in document_type.fields().values()])} 
            from 
            {document_type.CollectionName} {'' if len(kwargs) == 0 else f"where {' '.join([f'{key}=?' for key in keys])}"}
        """
        self.session.execute(read_sql, [MSSQLDatabase._field_where({f.name: f for f in document_type.fields().values()}[key], normalized_kwargs[key]) for key in keys])
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
                update 
                {document.CollectionName} 
                set 
                {','.join([f'{key}=?' for key in keys])} where object_id=?
            """
            self.session.execute(update_sql,
                                 [MSSQLDatabase._field_update({f.name: f for f in document.fields().values()}[key], dictionary[key]) for key in keys] + [str(document.object_id)])

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
