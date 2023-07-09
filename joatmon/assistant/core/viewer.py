from __future__ import print_function

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    @staticmethod
    def help():
        return {
            "name": "viewer",
            "description": "a function for user to view event of the iva",
            "parameters": {
                "type": "object",
                "properties": {
                },
                "required": []
            }
        }

    def run(self):
        if not self.stop_event.is_set():
            self.stop_event.set()
