from __future__ import print_function

import argparse
import sys
import time

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api=None):
        super(Task, self).__init__(api, False, 1, 100)

        parser = argparse.ArgumentParser()
        parser.add_argument('--message', type=str)

        namespace, _ = parser.parse_known_args(sys.argv)

        self.action = None
        if namespace.message:
            self.action = ['message', namespace.message]

    @staticmethod
    def help(api):
        message = """
        this module can be used to use openai api
            --ask to use chat-gpt
            --image to use dall-e
            --transcribe to use whisper
        """
        if api is not None:
            api.output(message)
            time.sleep(7)
        else:
            print(message)

    def run(self):
        try:
            self.api.output(self.action[1].replace('"', ''))

            if not self.event.is_set():
                self.event.set()
        except:
            ...


if __name__ == '__main__':
    Task(None).run()
