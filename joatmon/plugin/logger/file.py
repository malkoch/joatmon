import datetime
import json
import os

from joatmon.plugin.logger.core import LoggerPlugin


class FileLogger(LoggerPlugin):
    """
    FileLogger class that inherits from the LoggerPlugin class. It implements the abstract methods of the LoggerPlugin class
    using a file for logging operations.

    Attributes:
        level (str): The level of logging.
        folder (str): The base folder where the log files will be stored.
        language (str): The language for logging.
        ip (str): The IP address for logging.
    """

    def __init__(self, level: str, base_folder: str, language, ip):
        """
         Initialize FileLogger with the given level, base folder, language, and IP.

         Args:
             level (str): The level of logging.
             base_folder (str): The base folder where the log files will be stored.
             language (str): The language for logging.
             ip (str): The IP address for logging.
         """
        super(FileLogger, self).__init__(level, language, ip)

        self.folder = base_folder

    async def _write(self, log: dict):
        """
        Write a log to a file.

        This method writes the log to a file in the base folder. The file is named with the current date and the logs are appended to it.

        Args:
            log (dict): The log to be written.
        """
        dt = datetime.datetime.now()
        folder = os.path.join(self.folder, dt.strftime(f'%Y{os.sep}%m{os.sep}'))
        if not os.path.exists(folder):
            os.makedirs(folder)
        file = os.path.join(folder, dt.strftime(f'%d') + '.log')
        with open(file, 'a') as f:
            f.write(json.dumps(log) + os.linesep)
