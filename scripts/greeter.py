from __future__ import print_function

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    @staticmethod
    def params():
        return ['message']

    def run(self):
        message = self.kwargs.get('message', '')
        if not message:
            self.api.output('what do you want the message to be')
            message = self.api.input()

        self.api.say(message)

        self.event.set()


if __name__ == '__main__':
    Task(None).run()
