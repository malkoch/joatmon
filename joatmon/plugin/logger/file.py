import os
from datetime import datetime

from joatmon.plugin.logger.core import Logger


class FileLogger(Logger):
    def __init__(self, alias: str, level: str, connection: str):
        super(FileLogger, self).__init__(alias, level)

        self.path = connection

    def _write(self, log):
        date, time = datetime.utcnow().isoformat().split('T')
        year, month, day = date.split('-')
        folder_path = os.path.join(year, month)
        if not os.path.exists(os.path.join(self.path, folder_path)):
            os.makedirs(os.path.join(self.path, folder_path))

        file_path = os.path.join(self.path, folder_path, f'{day}.log')
        with open(file_path, 'a') as file:
            message = f'Level: {log["level"]} - Timestamp: {datetime.utcnow().isoformat()} - Message: {log}'
            file.write(message + '\n')
