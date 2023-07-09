from __future__ import print_function

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    @staticmethod
    def help():
        return {
            "name": "ls",
            "description": "a function for user to list the given directory, if path is not given it will list the current working directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "directory to list"
                    }
                },
                "required": []
            }
        }

    def run(self):
        parent_os_path = self.kwargs.get('parent_os_path', '')
        os_path = self.kwargs.get('os_path', '')

        self.api.say(os_path)

        if not self.stop_event.is_set():
            self.stop_event.set()
