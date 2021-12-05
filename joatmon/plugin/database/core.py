from joatmon.core import CoreException
from joatmon.database.query import QueryBuilder
from joatmon.plugin.core import Plugin


class DatabaseError(CoreException):
    pass


class DuplicateKeyError(DatabaseError):
    pass


class InvalidDocumentError(DatabaseError):
    pass


class Database(Plugin):
    def __init__(self, alias):
        super(Database, self).__init__(alias)

    def __enter__(self):
        self.start()
        return super(Database, self).__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.commit()
        else:
            self.abort()
        self.end()
        super(Database, self).__exit__(exc_type, exc_val, exc_tb)

    def start(self):
        raise NotImplementedError

    def commit(self):
        raise NotImplementedError

    def abort(self):
        raise NotImplementedError

    def end(self):
        raise NotImplementedError

    def up(self):  # migration, maybe need a version name
        ...

    def down(self):  # migration, maybe need a version name
        ...

    def load(self):  # migration, maybe need a version name
        ...

    def drop(self):
        raise NotImplementedError

    def initialize(self):
        raise NotImplementedError

    def save(self, *documents):
        raise NotImplementedError

    def read(self, document_type, **kwargs):
        raise NotImplementedError

    def update(self, *documents):
        raise NotImplementedError

    def delete(self, *documents):
        raise NotImplementedError

    def execute(self, query: QueryBuilder):
        raise NotImplementedError
