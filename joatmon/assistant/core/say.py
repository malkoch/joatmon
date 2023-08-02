from __future__ import print_function

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    @staticmethod
    def help():
        return {
            "name": "say",
            "description": "a function for user to make iva say something",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "the message to say"
                    }
                },
                "required": ["message"]
            }
        }

    def run(self):
        to_say = self.kwargs.get('message', '') or self.api.input('what do you want me to say')
        self.api.output(to_say)

        if not self.stop_event.is_set():
            self.stop_event.set()
