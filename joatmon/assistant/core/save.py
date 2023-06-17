from __future__ import print_function

import os

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    def run(self):
        path = self.kwargs.get('path', '') or self.api.listen('what is the path')
        message = self.kwargs.get('message', '') or self.api.listen('what is the message')

        parent_os_path = self.kwargs.get('parent_os_path', '')
        os_path = self.kwargs.get('os_path', '')

        p = path if os.path.isabs(path) else os.path.join(parent_os_path, os_path[1:], path)

        with open(p, 'wb') as file:
            file.write(message.encode('utf-8', errors='ignore'))

        if not self.event.is_set():
            self.event.set()


if __name__ == '__main__':
    Task(None).run()
