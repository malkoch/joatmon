from __future__ import print_function

from joatmon.assistant.task import BaseTask
from joatmon.search import summary


class Task(BaseTask):
    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    @staticmethod
    def help():
        return {
            "name": "search",
            "description": "a function for user to search an expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "message to search for"
                    }
                },
                "required": ["message"]
            }
        }

    def run(self):
        message = self.kwargs.get('message', '') or self.api.input('what do you want the message to be')

        result = summary(message)
        self.api.output(result)

        if not self.stop_event.is_set():
            self.stop_event.set()
