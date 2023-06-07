from __future__ import print_function

import os

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    @staticmethod
    def create(api):
        return {}

    def run(self):
        self.api.output('path: ')
        message = self.api.input()

        for item in os.listdir(message):
            self.api.output(item)

        if not self.event.is_set():
            self.event.set()


if __name__ == '__main__':
    Task(None).run()
