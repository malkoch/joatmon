from joatmon.plugin.core import Plugin


class DatabasePlugin(Plugin):
    """
    DatabasePlugin class that inherits from the Plugin class. It is an abstract class that provides
    the structure for database operations. The methods in this class should be implemented in the child classes.
    """

    # query can be a dictionary, query builder or formatted query
    # read write strategy
    # when reading, might read on cache so that the read operation will be reduced
    # when writing, might write to message queue and consume in batcher so that the write operation will be reduced
    # after writing the read cache should be updated as well
    async def __aenter__(self):
        """
        This method is called when the 'with' statement is used. It starts the database connection.

        Returns:
            self: The current instance of the class.
        """
        await self.start()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        This method is called when the 'with' statement is finished. It commits or aborts the transaction
        depending on whether an exception occurred or not. It then ends the database connection.

        Args:
            exc_type (type): The type of the exception that occurred, if any.
            exc_val (Exception): The instance of the exception that occurred, if any.
            exc_tb (traceback): A traceback object encapsulating the call stack at the point where the exception occurred, if any.

        Returns:
            self: The current instance of the class.

        Raises:
            exc_val: If an exception occurred during the 'with' statement.
        """
        if exc_type is None:
            await self.commit()
        else:
            await self.abort()
            raise exc_val
        await self.end()

        return self

    async def create(self, document):
        """
        This is an abstract method that should be implemented in the child classes. It is used to
        create a new document in the database.

        Args:
            document (dict): The document to be created.

        Raises:
            NotImplementedError: This method should be implemented in the child classes.
        """
        raise NotImplementedError

    async def alter(self, document):
        """
        This is an abstract method that should be implemented in the child classes. It is used to
        alter an existing document in the database.

        Args:
            document (dict): The document to be altered.

        Raises:
            NotImplementedError: This method should be implemented in the child classes.
        """
        raise NotImplementedError

    async def drop(self, document):
        """
        This is an abstract method that should be implemented in the child classes. It is used to
        drop a document from the database.

        Args:
            document (dict): The document to be dropped.

        Raises:
            NotImplementedError: This method should be implemented in the child classes.
        """
        raise NotImplementedError

    async def insert(self, document, *docs):
        """
        This is an abstract method that should be implemented in the child classes. It is used to
        insert one or more documents into the database.

        Args:
            document (dict): The first document to be inserted.
            *docs (dict): Additional documents to be inserted.

        Raises:
            NotImplementedError: This method should be implemented in the child classes.
        """
        raise NotImplementedError

    async def read(self, document, query):  # extra filters, limit, skip, sort etc.
        """
        This is an abstract method that should be implemented in the child classes. It is used to
        read a document from the database.

        Args:
            document (dict): The document to be read.
            query (dict): The query to be used for reading the document.

        Raises:
            NotImplementedError: This method should be implemented in the child classes.
        """
        raise NotImplementedError

    async def update(self, document, query, update):
        """
        This is an abstract method that should be implemented in the child classes. It is used to
        update a document in the database.

        Args:
            document (dict): The document to be updated.
            query (dict): The query to be used for updating the document.
            update (dict): The update to be applied to the document.

        Raises:
            NotImplementedError: This method should be implemented in the child classes.
        """
        raise NotImplementedError

    async def delete(self, document, query):
        """
        This is an abstract method that should be implemented in the child classes. It is used to
        delete a document from the database.

        Args:
            document (dict): The document to be deleted.
            query (dict): The query to be used for deleting the document.

        Raises:
            NotImplementedError: This method should be implemented in the child classes.
        """
        raise NotImplementedError

    async def view(self, document, query):
        """
        This is an abstract method that should be implemented in the child classes. It is used to
        view a document in the database.

        Args:
            document (dict): The document to be viewed.
            query (dict): The query to be used for viewing the document.

        Raises:
            NotImplementedError: This method should be implemented in the child classes.
        """
        raise NotImplementedError

    async def execute(self, document, query):
        """
        This is an abstract method that should be implemented in the child classes. It is used to
        execute a query on a document in the database.

        Args:
            document (dict): The document to be queried.
            query (dict): The query to be executed.

        Raises:
            NotImplementedError: This method should be implemented in the child classes.
        """
        raise NotImplementedError

    async def count(self, query):
        """
        This is an abstract method that should be implemented in the child classes. It is used to
        count the number of documents that match a query in the database.

        Args:
            query (dict): The query to be counted.

        Raises:
            NotImplementedError: This method should be implemented in the child classes.
        """
        raise NotImplementedError

    async def start(self):
        """
        This is an abstract method that should be implemented in the child classes. It is used to
        start a database transaction.

        Raises:
            NotImplementedError: This method should be implemented in the child classes.
        """
        raise NotImplementedError

    async def commit(self):
        """
        This is an abstract method that should be implemented in the child classes. It is used to
        commit a database transaction.

        Raises:
            NotImplementedError: This method should be implemented in the child classes.
        """
        raise NotImplementedError

    async def abort(self):
        """
        This is an abstract method that should be implemented in the child classes. It is used to
        abort a database transaction.

        Raises:
            NotImplementedError: This method should be implemented in the child classes.
        """
        raise NotImplementedError

    async def end(self):
        """
        This is an abstract method that should be implemented in the child classes. It is used to
        end a database transaction.

        Raises:
            NotImplementedError: This method should be implemented in the child classes.
        """
        raise NotImplementedError
