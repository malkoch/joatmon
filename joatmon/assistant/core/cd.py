from __future__ import print_function

import os

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    @staticmethod
    def help():
        return {
            "name": "cd",
            "description": "a function for user to change the current directory to given path",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "the path that the user want to change to"
                    }
                },
                "required": ["path"]
            }
        }

    def run(self):
        path = self.kwargs.get('path', None) or self.api.input('what is path that you want to change')

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

        self.api.output(f'current path is: {os_path}')

        if not self.stop_event.is_set():
            self.stop_event.set()
