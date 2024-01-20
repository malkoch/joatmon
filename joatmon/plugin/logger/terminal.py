import json

from joatmon.plugin.logger.core import LoggerPlugin


class TerminalLogger(LoggerPlugin):
    """
    TerminalLogger class that inherits from the LoggerPlugin class. It implements the abstract methods of the LoggerPlugin class
    using the terminal for logging operations.

    Attributes:
        level (str): The level of logging.
        language (str): The language for logging.
        ip (str): The IP address for logging.
    """

    def __init__(self, level: str, language, ip):
        """
        Initialize TerminalLogger with the given level, language, and IP.

        Args:
            level (str): The level of logging.
            language (str): The language for logging.
            ip (str): The IP address for logging.
        """
        super(TerminalLogger, self).__init__(level, language, ip)

    async def _write(self, log: dict):
        """
        Write a log to the terminal.

        This method prints the log to the terminal.

        Args:
            log (dict): The log to be written.
        """
        print(json.dumps(log))
