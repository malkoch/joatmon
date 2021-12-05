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
        parser.add_argument('--value', type=str)

        namespace, _ = parser.parse_known_args(sys.argv)

        self.config = None
        if namespace.create:
            self.config = {
                'action': 'create',
                'name': namespace.create,
                'value': namespace.value
            }
        elif namespace.update:
            self.config = {
                'action': 'update',
                'name': namespace.update,
                'value': namespace.value
            }
        elif namespace.delete:
            self.config = {
                'action': 'delete',
                'name': namespace.delete
            }

    def run(self):
        def set_config(parent, name, value):
            if name == '':
                return

            names = name.split('.')
            if names[0] not in parent:
                if len(names) == 1:
                    if value is not None:
                        parent[names[0]] = value
                    else:
                        del parent[names[0]]
                else:
                    parent[names[0]] = {}
            else:
                if len(names) == 1:
                    if value is not None:
                        parent[names[0]] = value
                    else:
                        del parent[names[0]]
            set_config(parent[names[0]], '.'.join(names[1:]), value)

        settings = json.loads(open('iva.json', 'r').read())
        configs = settings.get('configs', {})

        if self.config['action'] == 'create':
            set_config(configs, self.config['name'], self.config['value'])
            settings['configs'] = configs
            open('iva.json', 'w').write(json.dumps(settings, indent=4))
        elif self.config['action'] == 'update':
            set_config(configs, self.config['name'], self.config['value'])
            settings['configs'] = configs
            open('iva.json', 'w').write(json.dumps(settings, indent=4))
        elif self.config['action'] == 'delete':
            set_config(configs, self.config['name'], None)
            settings['configs'] = configs
            open('iva.json', 'w').write(json.dumps(settings, indent=4))
        else:
            raise ValueError(f'arguments are not recognized')

        if not self.event.is_set():
            self.event.set()


if __name__ == '__main__':
    Task(None).run()
