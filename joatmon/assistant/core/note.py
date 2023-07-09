from __future__ import print_function

from joatmon.assistant.service import BaseService
from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    @staticmethod
    def help():
        return {
            "name": "note",
            "description": "a function for user to take note and list them afterwards",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["create", "list"]
                    }
                },
                "required": ["mode"]
            }
        }

    def run(self):
        mode = self.kwargs.get('mode', None)

        if mode == 'list':
            print(self.kwargs.get('mode', None), self.kwargs.get('event', None), self.kwargs.get('date', None))
        if mode == 'create':
            print(self.kwargs.get('mode', None), self.kwargs.get('event', None), self.kwargs.get('date', None))


class Service(BaseService):
    def __init__(self, api, **kwargs):
        super(Service, self).__init__(api, **kwargs)

    @staticmethod
    def help():
        return {}

    def run(self):
        ...
