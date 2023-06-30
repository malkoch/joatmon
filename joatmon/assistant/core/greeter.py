from __future__ import print_function

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    @staticmethod
    def help():
        return ''

    @staticmethod
    def params():
        return ['message']

    def run(self):
        message = self.kwargs.get('message', '') or self.api.listen('what do you want the message to be')
        self.api.say(message)
        self.event.set()


if __name__ == '__main__':
    Task(None).run()
