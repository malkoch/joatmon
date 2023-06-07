from __future__ import print_function

import os

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api, *args, **kwargs):
        super(Task, self).__init__(api, *args, **kwargs)

    @staticmethod
    def create(api):
        return {}

    def run(self):
        path = self.args[0]

        parent_os_path = self.kwargs.get('parent_os_path', '')
        os_path = self.kwargs.get('os_path', '')

        match path:
            case '.':
                ...
            case '..':
                if os_path != os.sep:
                    os_path = os.sep.join(os_path.split(os.sep)[:-1])
                    if os_path == '':
                        os_path = os.sep
            case _:
                os_path = os.path.join(os_path, path)

        self.api.os_path = os_path

        if not self.event.is_set():
            self.event.set()


if __name__ == '__main__':
    Task(None).run()
