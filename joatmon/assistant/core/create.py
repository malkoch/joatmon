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
        return {
            "name": "create",
            "description": "a function for user to create service or task",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["task", "service"]
                    }
                },
                "required": ["mode"]
            }
        }

    def run(self):
        if self.kwargs.get('mode', '') == 'task':
            task.create(self.api)
        if self.kwargs.get('mode', '') == 'service':
            service.create(self.api)

        if not self.stop_event.is_set():
            self.stop_event.set()
