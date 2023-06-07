from __future__ import print_function

import json

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api, *args, **kwargs):
        super(Task, self).__init__(api, *args, **kwargs)

    @staticmethod
    def params():
        return ['mode']

    def run(self):
        mode = self.kwargs.get('mode', '')

        contacts = json.loads(open('iva.json', 'r').read()).get('contacts', [])

        if mode == 'list':
            ...
        elif mode == 'create':
            ...

        if not self.event.is_set():
            self.event.set()


if __name__ == '__main__':
    Task(None).run()
