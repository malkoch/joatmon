import dataclasses
import enum
import importlib.util
import json
import os
import threading

from transitions import Machine

from joatmon.core.utility import (
    first,
    JSONEncoder
)


class BaseService:
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
        self.name = name
        self.api = api
        self.kwargs = kwargs
        self.event = threading.Event()
        self.thread = threading.Thread(target=self.run)

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
        return not self.event.is_set()

    def start(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.thread.start()

    def stop(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.event.set()


class ServiceState(enum.Enum):
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
    stopped = enum.auto()


# create from json and to json methods
@dataclasses.dataclass
class ServiceInfo:
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
    state: ServiceState
    service: BaseService


def on_begin(name, *args, **kwargs):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    ...


def on_error(name, *args, **kwargs):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    # if the service has recovery option, run it
    # else end
    ...


def on_end(name, *args, **kwargs):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    ...


def create(api):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    name = api.input('name')
    priority = api.input('priority')
    mode = api.input('mode')
    script = api.input('script')
    kwargs = {}
    for k in get_class(script).params():
        kwargs[k] = api.input(k)

    # on error
    # on recovery
    create_args = {'name': name, 'priority': priority, 'mode': mode, 'script': script, 'status': True, 'kwargs': kwargs}

    settings = json.loads(open(os.path.join(os.environ.get('ASSISTANT_HOME'), 'system.json'), 'r').read())
    services = settings.get('services', [])

    services.append(create_args)

    settings['services'] = services
    open(os.path.join(api.base, 'iva.json'), 'w').write(json.dumps(settings, indent=4, cls=JSONEncoder))


def get_class(name):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    service = None

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
            except ModuleNotFoundError:
                continue

            action_module = getattr(_module, name, None)

        if action_module is None:
            continue

        service = getattr(action_module, 'Service', None)

    return service


def get(api, name):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    settings = json.loads(open(os.path.join(os.environ.get('ASSISTANT_HOME'), 'system.json'), 'r').read())
    task_info = first(filter(lambda x: x['status'] and x['name'] == name, settings.get('services', [])))

    if task_info is None:
        task_info = {'script': name, 'kwargs': {}}

    script = task_info['script']

    service = get_class(script)

    if service is None:
        api.output('service is not found')
        return None

    kwargs = {**task_info.get('kwargs', {}), 'base': os.environ.get('ASSISTANT_HOME')}

    service = service(name, api, **kwargs)
    return service
