from __future__ import print_function

import argparse
import sys

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api):
        super(Task, self).__init__(api, False, 1, 100)

        parser = argparse.ArgumentParser()
        parser.add_argument('--args', type=str)

        namespace, _ = parser.parse_known_args(sys.argv)

        self.action = [namespace.args]

    @staticmethod
    def help(api):
        ...

    def run(self):
        print(f'dummy task is running {self.action}')

        if not self.event.is_set():
            self.event.set()


if __name__ == '__main__':
    Task(None).run()
