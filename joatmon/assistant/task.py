import dataclasses
import enum
import importlib.util
import json
import os
import threading

from transitions import Machine

from joatmon.core.event import Event
from joatmon.core.utility import (JSONEncoder, first, get_module_classes)


class BaseTask:
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

    def __init__(self, name, **kwargs):  # another parameter called cache output
        self.name = name
        self.kwargs = kwargs
        self.thread = threading.Thread(target=self.run)
        self.stop_event = threading.Event()

        states = ['none', 'started', 'stopped', 'running', 'exception', 'starting', 'stopping']
        self.machine = Machine(model=self, states=states, initial='none')

    @staticmethod
    def help():
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return {}

    def run(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        raise NotImplementedError

    def running(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return not self.stop_event.is_set()

    def start(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        events['begin'].fire()

        self.thread.start()

    def stop(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.stop_event.set()

        events['end'].fire()


class TaskState(enum.Enum):
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

    running = enum.auto()
    finished = enum.auto()


# create from json and to json methods
@dataclasses.dataclass
class TaskInfo:
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

    name: str
    state: TaskState
    task: BaseTask


events = {
    'begin': Event(),
    'end': Event(),
    'error': Event(),
    'create': Event(),
    'update': Event(),
    'delete': Event(),
}


def create(name, priority, on, script, kwargs):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    # need last run time
    # need next run time
    # need last run result
    # need interval
    # if on == 'interval' need to ask for interval as well
    create_args = {'name': name, 'priority': priority, 'on': on, 'script': script, 'status': True, 'kwargs': kwargs}

    settings = json.loads(open(os.path.join(os.environ.get('ASSISTANT_HOME'), 'system.json'), 'r').read())
    tasks = settings.get('tasks', [])

    tasks.append(create_args)

    settings['tasks'] = tasks
    open(os.path.join(os.environ.get('ASSISTANT_HOME'), 'system.json'), 'w').write(json.dumps(settings, indent=4, cls=JSONEncoder))


def get_class(name):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    task = None

    settings = json.loads(open(os.path.join(os.environ.get('ASSISTANT_HOME'), 'system.json'), 'r').read())
    for scripts in settings.get('scripts', []):
        if os.path.isabs(scripts):
            if os.path.exists(scripts) and os.path.exists(os.path.join(scripts, f'{name}.py')):
                spec = importlib.util.spec_from_file_location(scripts, os.path.join(scripts, f'{name}.py'))
                action_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(action_module)
            else:
                continue
        else:
            try:
                _module = __import__(scripts, fromlist=[f'{name}'])
            except ModuleNotFoundError as ex:
                print(str(ex))
                continue

            action_module = getattr(_module, name, None)

        if action_module is None:
            continue

        for class_ in get_module_classes(action_module):
            if not issubclass(class_[1], BaseTask) or class_[1] is BaseTask:
                continue

            task = class_[1]

    return task


def get(name, kwargs):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    settings = json.loads(open(os.path.join(os.environ.get('ASSISTANT_HOME'), 'system.json'), 'r').read())
    task_info = first(filter(lambda x: x['status'] and x['name'] == name, settings.get('tasks', [])))

    if task_info is None:
        task_info = {'script': name, 'kwargs': {}}

    script = task_info['script']

    task = get_class(script)

    if task is None:
        return None

    kwargs = {**(kwargs or {}), **task_info.get('kwargs', {})}

    task = task(name, **kwargs)

    return task
