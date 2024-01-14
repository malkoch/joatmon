import importlib.util
import json
import os

from joatmon.assistant.task import BaseTask
from joatmon.core.utility import JSONEncoder, get_module_classes


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

    def __init__(self, name, **kwargs):
        super(Task, self).__init__(name, **kwargs)

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
        script = self.kwargs.get('script', None)

        tasks = []

        settings = json.loads(open(os.path.join(os.environ.get('ASSISTANT_HOME'), 'system.json'), 'r').read())
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

                    if script is not None and action_module != script:
                        continue

                    task = getattr(action_module, 'Task', None)
                    if task is None:
                        continue
                    tasks.append(task)
            else:
                try:
                    _module = __import__('.'.join(scripts.split('.')), fromlist=[scripts.split('.')[-1]])

                    scripts = _module.__path__[0]

                    for script_file in os.listdir(scripts):
                        if '__' in script_file:
                            continue

                        if script is not None and script_file.replace('.py', '') != script:
                            continue

                        spec = importlib.util.spec_from_file_location(
                            scripts, os.path.join(scripts, script_file)
                        )
                        action_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(action_module)

                        for class_ in get_module_classes(action_module):
                            if not issubclass(class_[1], BaseTask) or class_[1] is BaseTask:
                                continue

                            tasks.append(class_[1])
                except ModuleNotFoundError:
                    print('module not found')
                    continue

        functions = list(map(lambda x: x.help(), tasks))
        functions = list(filter(lambda x: x, functions))
        _ = json.dumps(functions, indent=4, cls=JSONEncoder)

        if not self.stop_event.is_set():
            self.stop_event.set()
