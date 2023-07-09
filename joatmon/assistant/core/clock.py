from __future__ import print_function

import datetime

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    @staticmethod
    def help():
        return {
            "name": "clock",
            "description": "a function for user to learn current time and date",
            "parameters": {
                "type": "object",
                "properties": {
                },
                "required": []
            }
        }

    def run(self):
        self.api.say(datetime.datetime.now().isoformat())

        if not self.stop_event.is_set():
            self.stop_event.set()
