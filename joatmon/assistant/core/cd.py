from __future__ import print_function

import os

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
        if (path := self.kwargs.get('path', None)) is None:
            self.api.say('what is path that you want to change')
            path = self.api.listen()

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

        self.api.say(f'current path is: {os_path}')

        if not self.event.is_set():
            self.event.set()


if __name__ == '__main__':
    Task(None).run()
