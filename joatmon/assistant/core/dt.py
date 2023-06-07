from __future__ import print_function

import datetime

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api, *args, **kwargs):
        super(Task, self).__init__(api, *args, **kwargs)

    @staticmethod
    def create(api):
        return {}

    def run(self):
        self.api.show_('t1', 'datetime', [datetime.datetime.now().isoformat()])

        if not self.event.is_set():
            self.event.set()


if __name__ == '__main__':
    Task(None).run()
