from __future__ import print_function

import os
import shutil

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    @staticmethod
    def help():
        return ''

    @staticmethod
    def params():
        return []

    def run(self):
        old_file = self.kwargs.get('old_file', '') or self.api.listen('what is the old file')
        new_file = self.kwargs.get('new_file', '') or self.api.listen('what is the new file')

        parent_os_path = self.kwargs.get('parent_os_path', '')
        os_path = self.kwargs.get('os_path', '')

        shutil.move(os.path.join(parent_os_path, os_path[1:], old_file), os.path.join(parent_os_path, os_path[1:], new_file))

        if not self.event.is_set():
            self.event.set()


if __name__ == '__main__':
    Task(None).run()
