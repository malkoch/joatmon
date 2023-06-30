from __future__ import print_function

import datetime

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    @staticmethod
    def help():
        return ''

    @staticmethod
    def params():
        return []

    def run(self):
        self.api.show_('t1', 'datetime', [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')])

        if not self.event.is_set():
            self.event.set()


if __name__ == '__main__':
    Task(None).run()
