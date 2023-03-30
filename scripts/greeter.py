from __future__ import print_function

import time

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    arguments = {
        'message': 'message to greet with'
    }

    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    def run(self):
        message = self.kwargs.get('message', '')
        if message:
            self.api.output(message)

        if not self.event.is_set():
            self.event.set()


if __name__ == '__main__':
    Task(None).run()
