from __future__ import print_function

import argparse
import json
import sys

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api=None):
        super(Task, self).__init__(api, False, 1, 100)

        parser = argparse.ArgumentParser()
        parser.add_argument('--create', type=str)
        parser.add_argument('--update', type=str)
        parser.add_argument('--delete', type=str)
        parser.add_argument('--command', type=str)
        parser.add_argument('--priority', type=int)

        namespace, _ = parser.parse_known_args(sys.argv)

        self.action = None
        if namespace.create:
            self.action = ['create', namespace.create, namespace.command]
        elif namespace.update:
            self.action = ['update', namespace.update, namespace.command]
        elif namespace.delete:
            self.action = ['delete', namespace.delete, namespace.command]

    def run(self):
        json.loads(open('iva.json', 'r').read())

        if self.action[0] == 'create':
            ...
        elif self.action[0] == 'update':
            ...
        elif self.action[0] == 'delete':
            ...
        else:
            raise ValueError(f'arguments are not recognized')

        if not self.event.is_set():
            self.event.set()


if __name__ == '__main__':
    Task(None).run()
