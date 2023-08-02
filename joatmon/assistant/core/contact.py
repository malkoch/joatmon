from __future__ import print_function

import json

from joatmon.assistant.task import BaseTask
from joatmon.utility import JSONEncoder


class Task(BaseTask):
    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    @staticmethod
    def help():
        return {
            "name": "contact",
            "description": "a function for user to list, create, update, search delete contact",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["list", "create", "update", "search", "delete"]
                    },
                    "name": {
                        "type": "string",
                        "description": "name of the contact"
                    },
                    "email": {
                        "type": "string",
                        "description": "email address of the contact"
                    },
                    "phone": {
                        "type": "string",
                        "description": "phone number of the contact"
                    }
                },
                "required": ["mode"]
            }
        }

    def run(self):
        mode = self.kwargs.get('mode', '')

        settings = json.loads(open('iva/iva.json', 'r').read())
        contacts = settings.get('contacts', [])

        if mode == 'list':
            ...
        elif mode == 'create':
            name = self.kwargs.get('name', None) or self.api.input('what is the name of the contact')
            email = self.kwargs.get('email', None) or self.api.input('what is the email')
            phone = self.kwargs.get('phone', None) or self.api.input('what is the phone number')

            contacts.append({'name': name, 'email': email, 'phone': phone})
        elif mode == 'update':
            ...
        elif mode == 'search':
            ...

        settings['contacts'] = contacts
        open('iva/iva.json', 'w').write(json.dumps(settings, indent=4, cls=JSONEncoder))

        if not self.stop_event.is_set():
            self.stop_event.set()
