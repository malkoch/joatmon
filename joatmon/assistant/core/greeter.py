from __future__ import print_function

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    @staticmethod
    def help():
        return {
            "name": "weather",
            "description": "a function for user to greet someone",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "message to greet someone"
                    }
                },
                "required": ["message"]
            }
        }

    def run(self):
        message = self.kwargs.get('message', '') or self.api.input('what do you want the message to be')
        self.api.output(message)
        self.stop_event.set()
