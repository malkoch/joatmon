from __future__ import print_function

import subprocess

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    @staticmethod
    def help():
        return {
            "name": "run",
            "description": "a function for user to run an executable",
            "parameters": {
                "type": "object",
                "properties": {
                    "executable": {
                        "type": "string",
                        "description": "executable to run"
                    }
                },
                "required": ["executable"]
            }
        }

    def run(self):
        executable = self.kwargs.get('executable', '') or self.api.input('what do you want to run')

        subprocess.run(['python.exe', executable])

        if not self.stop_event.is_set():
            self.stop_event.set()
