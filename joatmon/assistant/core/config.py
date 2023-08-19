import json
import os.path

from joatmon.assistant.task import BaseTask
from joatmon.utility import JSONEncoder


class Task(BaseTask):
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __init__(self, name, api, **kwargs):
        super(Task, self).__init__(name, api, **kwargs)

    @staticmethod
    def help():
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return {
            'name': 'config',
            'description': 'a function for user to configure the iva',
            'parameters': {
                'type': 'object',
                'properties': {
                    'action': {'type': 'string', 'enum': ['create', 'update', 'delete']},
                    'name': {'type': 'string', 'description': 'name of the configuration'},
                    'value': {'type': 'string', 'description': 'value of the configuration'},
                },
                'required': ['action', 'name'],
            },
        }

    def run(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        action = self.kwargs.get('action', '') or self.api.input('what do you want to configure')
        assert action in ('create', 'update', 'delete')
        name = self.kwargs.get('name', '') or self.api.input('what is the name')
        value = self.kwargs.get('value', '') or self.api.input('what is the value')

        cfg = {'action': action, 'name': name, 'value': value}

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

        settings = json.loads(open(os.path.join(os.environ.get('IVA_PATH'), 'iva.json'), 'r').read())
        config = settings.get('config', {})

        if cfg['action'] == 'create':
            set_config(config, cfg['name'], cfg['value'])
            settings['config'] = config
            open(os.path.join(os.environ.get('IVA_PATH'), 'iva.json'), 'w').write(json.dumps(settings, indent=4, cls=JSONEncoder))
        elif cfg['action'] == 'update':
            set_config(config, cfg['name'], cfg['value'])
            settings['config'] = config
            open(os.path.join(os.environ.get('IVA_PATH'), 'iva.json'), 'w').write(json.dumps(settings, indent=4, cls=JSONEncoder))
        elif cfg['action'] == 'delete':
            set_config(config, cfg['name'], None)
            settings['config'] = config
            open(os.path.join(os.environ.get('IVA_PATH'), 'iva.json'), 'w').write(json.dumps(settings, indent=4, cls=JSONEncoder))
        else:
            raise ValueError(f'arguments are not recognized')

        if not self.stop_event.is_set():
            self.stop_event.set()
