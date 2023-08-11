from __future__ import print_function

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    @staticmethod
    def help():
        return {
            "name": "camera",
            "description": "a function for user to take a photo of themselves",
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
