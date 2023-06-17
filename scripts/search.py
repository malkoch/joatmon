from __future__ import print_function

from joatmon.assistant.task import BaseTask
from joatmon.search import summary


class Task(BaseTask):
    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    @staticmethod
    def params():
        return []

    def run(self):
        message = self.kwargs.get('message', '') or self.api.listen('what do you want the message to be')

        result = summary(message)
        self.api.output(result)

        if not self.event.is_set():
            self.event.set()


if __name__ == '__main__':
    Task(None).run()
