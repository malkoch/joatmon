import datetime
import json
import os

from joatmon.plugin.logger.core import LoggerPlugin


class FileLogger(LoggerPlugin):
    def __init__(self, level: str, base_folder: str, language, ip):
        super(FileLogger, self).__init__(level, language, ip)

        self.folder = base_folder

    async def _write(self, log: dict):
        dt = datetime.datetime.now()
        folder = os.path.join(self.folder, dt.strftime(f'%Y{os.sep}%m{os.sep}'))
        if not os.path.exists(folder):
            os.makedirs(folder)
        file = os.path.join(folder, dt.strftime(f'%d') + '.log')
        with open(file, 'a') as f:
            f.write(json.dumps(log) + os.linesep)
