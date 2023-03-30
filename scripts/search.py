from __future__ import print_function

from joatmon.assistant.task import BaseTask
from joatmon.search.wikipedia import summary


class Task(BaseTask):
    run_arguments = {
        'message': ''
    }

    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    def run(self):
        result = summary(self.kwargs['message'])
        self.api.output(result)

        if not self.event.is_set():
            self.event.set()


if __name__ == '__main__':
    Task(None).run()
