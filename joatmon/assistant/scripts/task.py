from __future__ import print_function

import argparse
import json
import sys

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api=None):
        super(Task, self).__init__(api, False, 1, 100)

        parser = argparse.ArgumentParser()
        parser.add_argument('--create', type=str)
        parser.add_argument('--update', type=str)
        parser.add_argument('--delete', type=str)
        parser.add_argument('--command', type=str)
        parser.add_argument('--priority', type=int)
        parser.add_argument('--schedule', type=str)
        parser.add_argument('--status', type=bool)

        namespace, _ = parser.parse_known_args(sys.argv)

        self.action = None
        if namespace.create:
            self.action = {
                'action': 'create',
                'name': namespace.create,
                'command': namespace.command.replace('"', ''),
                'priority': namespace.priority,
                'schedule': namespace.schedule
            }
        elif namespace.update:
            self.action = {
                'action': 'update',
                'name': namespace.update,
                'command': namespace.command.replace('"', ''),
                'priority': namespace.priority,
                'schedule': namespace.schedule,
                'status': namespace.status
            }
        elif namespace.delete:
            self.action = {
                'action': 'delete',
                'name': namespace.delete
            }

    def run(self):
        settings = json.loads(open('iva.json', 'r').read())
        tasks = settings.get('tasks', {})

        if self.action['action'] == 'create':
            task = tasks.get(self.action['name'], None)
            if task:
                raise ValueError('task is already exists')
            task = {
                'command': self.action['command'],
                'priority': self.action['priority'] or 1,
                'schedule': self.action['schedule'] or '* * * * *',
                'status': True
            }
            tasks[self.action['name']] = task
            settings['tasks'] = tasks
            open('iva.json', 'w').write(json.dumps(settings, indent=4))
        elif self.action['action'] == 'update':
            task = tasks.get(self.action['name'], None)
            if task is None:
                raise ValueError('task does not exists')
            task = {
                'command': self.action['command'],
                'priority': self.action['priority'] or 1,
                'schedule': self.action['schedule'] or '* * * * *',
                'status': self.action['status'] or False,
            }
            tasks[self.action['name']] = task
            settings['tasks'] = tasks
            open('iva.json', 'w').write(json.dumps(settings, indent=4))
        elif self.action['action'] == 'delete':
            task = tasks.get(self.action['name'], None)
            if task is None:
                raise ValueError('task does not exists')
            del tasks[self.action['name']]
            settings['tasks'] = tasks
            open('iva.json', 'w').write(json.dumps(settings, indent=4))
        else:
            raise ValueError(f'arguments are not recognized')

        if not self.event.is_set():
            self.event.set()


if __name__ == '__main__':
    Task(None).run()
