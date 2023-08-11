from __future__ import print_function

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    @staticmethod
    def help():
        return {
            "name": "cwd",
            "description": "a function for user to learn current working directory",
            "parameters": {
                "type": "object",
                "properties": {
                },
                "required": []
            }
        }

    def run(self):
        parent_os_path = self.kwargs.get('parent_os_path', '')
        os_path = self.kwargs.get('os_path', '')

        self.api.output(os_path)

        if not self.stop_event.is_set():
            self.stop_event.set()
