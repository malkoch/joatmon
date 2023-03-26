from __future__ import print_function

import argparse
import sys

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api=None):
        super(Task, self).__init__(api)

        parser = argparse.ArgumentParser()
        parser.add_argument('--url', type=str)
        parser.add_argument('--file', type=str)
        parser.add_argument('--chunks', type=int)
        parser.add_argument('--resume', type=bool)

        namespace, _ = parser.parse_known_args(sys.argv)
        self.download = {
            'url': namespace.url,
            'file': namespace.file,
            'chunks': namespace.chunks,
            'resume': namespace.resume,
        }

    @staticmethod
    def help(api):
        ...

    def run(self):
        try:
            download(self.download['url'], self.download['file'], self.download['chunks'] or 10, self.download['resume'] or False)

            if not self.event.is_set():
                self.event.set()
        except:
            ...


if __name__ == '__main__':
    Task(None).run()
