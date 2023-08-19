import importlib.util
import inspect
import json
import os

from joatmon.assistant.task import BaseTask


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
            'name': 'help',
            'description': 'a function for user to learn about a function',
            'parameters': {
                'type': 'object',
                'properties': {
                    'function': {
                        'type': 'string',
                        'description': 'name of the function that the user want to learn about',
                    }
                },
                'required': ['name'],
            },
        }

    def run(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        settings = json.loads(open(os.path.join(os.environ.get('IVA_PATH'), 'iva.json'), 'r').read())
        for scripts in settings.get('scripts', []):
            if os.path.isabs(scripts) and os.path.exists(scripts):
                for module in list(
                        filter(
                            lambda x: '__' not in x,
                            map(lambda x: x.replace('.py', ''), os.listdir(scripts.replace('.', '/'))),
                        )
                ):
                    spec = importlib.util.spec_from_file_location(module, os.path.join(scripts, f'{module}.py'))
                    action_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(action_module)

                    task = getattr(action_module, 'Task', None)
                    if task is None:
                        continue
                    task.help(self)
            else:
                _module = __import__(scripts, fromlist=[''])

                for module in inspect.getmembers(_module, predicate=inspect.ismodule):
                    action_module = getattr(_module, module[0])

                    task = getattr(action_module, 'Task', None)
                    if task is None:
                        continue
                    task.help(self)

        if not self.stop_event.is_set():
            self.stop_event.set()
