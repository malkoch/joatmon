from joatmon.core import context
from joatmon.plugin.logger.core import LoggerPlugin


class DatabaseLogger(LoggerPlugin):
    """
    DatabaseLogger class that inherits from the LoggerPlugin class. It implements the abstract methods of the LoggerPlugin class
    using a database for logging operations.

    Attributes:
        level (str): The level of logging.
        database (str): The name of the database to be used for logging.
        cls (str): The class of the documents to be logged.
        language (str): The language for logging.
        ip (str): The IP address for logging.
    """

    def __init__(self, level: str, database: str, cls, language, ip):
        """
        Initialize DatabaseLogger with the given level, database, class, language, and IP.

        Args:
            level (str): The level of logging.
            database (str): The name of the database to be used for logging.
            cls (str): The class of the documents to be logged.
            language (str): The language for logging.
            ip (str): The IP address for logging.
        """
        super(DatabaseLogger, self).__init__(level, language, ip)

        self.database = database
        self.cls = cls

    async def _write(self, log: dict):
        """
        Write a log to the database.

        This method inserts the log into the database.

        Args:
            log (dict): The log to be written.
        """
        database = context.get_value(self.database)
        await database.insert(self.cls, log)
