import json

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    @staticmethod
    def help():
        return ''

    @staticmethod
    def params():
        return []

    def run(self):
        action = self.kwargs.get('action', '') or self.api.listen('what do you want to configure')
        assert action in ('create', 'update', 'delete')
        name = self.kwargs.get('name', '') or self.api.listen('what is the name')
        value = self.kwargs.get('value', '') or self.api.listen('what is the value')

        config = {
            'action': action,
            'name': name,
            'value': value
        }

        def set_config(_parent, _name, _value):
            if _name == '':
                return

            names = _name.split('.')
            if names[0] not in _parent:
                if len(names) == 1:
                    if _value is not None:
                        _parent[names[0]] = _value
                    else:
                        del _parent[names[0]]
                else:
                    _parent[names[0]] = {}
            else:
                if len(names) == 1:
                    if _value is not None:
                        _parent[names[0]] = _value
                    else:
                        del _parent[names[0]]
            set_config(_parent[names[0]], '.'.join(names[1:]), _value)

        settings = json.loads(open('iva.json', 'r').read())
        configs = settings.get('configs', {})

        if config['action'] == 'create':
            set_config(configs, config['name'], config['value'])
            settings['configs'] = configs
            open('iva.json', 'w').write(json.dumps(settings, indent=4))
        elif config['action'] == 'update':
            set_config(configs, config['name'], config['value'])
            settings['configs'] = configs
            open('iva.json', 'w').write(json.dumps(settings, indent=4))
        elif config['action'] == 'delete':
            set_config(configs, config['name'], None)
            settings['configs'] = configs
            open('iva.json', 'w').write(json.dumps(settings, indent=4))
        else:
            raise ValueError(f'arguments are not recognized')

        if not self.event.is_set():
            self.event.set()
