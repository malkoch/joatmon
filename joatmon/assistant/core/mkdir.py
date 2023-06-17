from __future__ import print_function

import os

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    def run(self):
        path = self.kwargs.get('path', '') or self.api.listen('what is the new folder name')

        parent_os_path = self.kwargs.get('parent_os_path', '')
        os_path = self.kwargs.get('os_path', '')

        if path.isalpha():
            os.mkdir(os.path.join(parent_os_path, os_path[1:], path))

        if not self.event.is_set():
            self.event.set()


if __name__ == '__main__':
    Task(None).run()
