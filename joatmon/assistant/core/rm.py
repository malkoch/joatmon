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

        if os.path.isdir(os.path.join(parent_os_path, os_path[1:], path)):
            os.removedirs(os.path.join(parent_os_path, os_path[1:], path))
        else:
            os.remove(os.path.join(parent_os_path, os_path[1:], path))

        if not self.event.is_set():
            self.event.set()


if __name__ == '__main__':
    Task(None).run()
