from __future__ import print_function

import os

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    @staticmethod
    def help():
        return {
            "name": "mkdir",
            "description": "a function for user to create a new directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "name of the new directory"
                    }
                },
                "required": ["path"]
            }
        }

    def run(self):
        path = self.kwargs.get('path', '') or self.api.listen('what is the new folder name')

        parent_os_path = self.kwargs.get('parent_os_path', '')
        os_path = self.kwargs.get('os_path', '')

        if path.isalpha():
            os.mkdir(os.path.join(parent_os_path, os_path[1:], path))

        if not self.stop_event.is_set():
            self.stop_event.set()
