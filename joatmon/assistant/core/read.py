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

        with open(os.path.join(parent_os_path, os_path[1:], path), 'rb') as file:
            self.api.output(file.read().decode('utf-8', errors='ignore'))

        if not self.event.is_set():
            self.event.set()


if __name__ == '__main__':
    Task(None).run()
