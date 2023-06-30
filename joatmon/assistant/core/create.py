from joatmon.assistant import (
    service,
    task
)
from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    @staticmethod
    def help():
        return ''

    @staticmethod
    def params():
        return ['mode']

    def run(self):
        if self.kwargs.get('mode', '') == 'task':
            task.create(self.api)
        if self.kwargs.get('mode', '') == 'service':
            service.create(self.api)

        if not self.event.is_set():
            self.event.set()
