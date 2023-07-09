from __future__ import print_function

from joatmon.assistant.service import BaseService
from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    @staticmethod
    def help():
        return {
            "name": "agenda",
            "description": "a function for user to create, list, delete, update, search an event in their agenda",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["create", "list", "delete", "update", "search"]
                    },
                    "event": {
                        "type": "string",
                        "description": "the event name"
                    },
                    "date": {
                        "type": "string",
                        "description": "datetime of the event"
                    }
                },
                "required": ["mode", "event"]
            }
        }

    def run(self):
        mode = self.kwargs.get('mode', None)

        if mode == 'list':
            print(self.kwargs.get('mode', None), self.kwargs.get('event', None), self.kwargs.get('date', None))
        if mode == 'create':
            print(self.kwargs.get('mode', None), self.kwargs.get('event', None), self.kwargs.get('date', None))
        if mode == 'delete':
            print(self.kwargs.get('mode', None), self.kwargs.get('event', None), self.kwargs.get('date', None))
        if mode == 'update':
            print(self.kwargs.get('mode', None), self.kwargs.get('event', None), self.kwargs.get('date', None))
        if mode == 'search':
            print(self.kwargs.get('mode', None), self.kwargs.get('event', None), self.kwargs.get('date', None))


class Service(BaseService):
    def __init__(self, api, **kwargs):
        super(Service, self).__init__(api, **kwargs)

    @staticmethod
    def help():
        return {}

    def run(self):
        ...
