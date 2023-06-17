from __future__ import print_function

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    def run(self):
        to_say = self.kwargs.get('message', '') or self.api.listen('what do you want me to say')
        self.api.say(to_say)

        if not self.event.is_set():
            self.event.set()


if __name__ == '__main__':
    Task(None).run()
