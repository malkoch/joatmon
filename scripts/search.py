from __future__ import print_function

from joatmon.assistant.task import BaseTask
from joatmon.search import summary


class Task(BaseTask):
    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    @staticmethod
    def create(api):
        api.output('what do you want the message to be')
        message = api.input()
        return {'message': message}

    def run(self):
        message = self.kwargs.get('message', '')
        if not message:
            self.api.output('what do you want the message to be')
            message = self.api.input()

        result = summary(message)
        self.api.output(result)

        if not self.event.is_set():
            self.event.set()


if __name__ == '__main__':
    Task(None).run()
