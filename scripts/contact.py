from __future__ import print_function

import json

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api, *args, **kwargs):
        super(Task, self).__init__(api, *args, **kwargs)

    @staticmethod
    def params():
        return ['mode']

    def run(self):
        mode = self.kwargs.get('mode', '')

        settings = json.loads(open('iva.json', 'r').read())
        contacts = settings.get('contacts', [])

        if mode == 'list':
            ...
        elif mode == 'create':
            name = self.api.listen('what is the name of the contact')
            email = self.api.listen('what is the email')
            phone = self.api.listen('what is the phone number')

            contacts.append({'name': name, 'email': email, 'phone': phone})
        elif mode == 'update':
            ...
        elif mode == 'search':
            ...

        settings['contacts'] = contacts
        open('iva.json', 'w').write(json.dumps(settings, indent=4))

        if not self.event.is_set():
            self.event.set()


if __name__ == '__main__':
    Task(None).run()
