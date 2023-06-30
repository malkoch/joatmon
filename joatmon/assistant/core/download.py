from __future__ import print_function

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, True, **kwargs)

    @staticmethod
    def help():
        return ''

    @staticmethod
    def params():
        return []

    def run(self):
        # need to do this in background
        # after it is done, need to notify user and prompt action to continue
        # it should not interfere with the current task that the user running
        if not self.event.is_set():
            self.event.set()


if __name__ == '__main__':
    Task(None).run()
