import dataclasses
import enum
import importlib.util
import json
import os
import threading

from transitions import Machine

from joatmon.utility import (
    first,
    JSONEncoder
)


class BaseService:
    def __init__(self, api, **kwargs):
        self.api = api
        self.kwargs = kwargs
        self.event = threading.Event()
        self.thread = threading.Thread(target=self.run)

        states = ['none', 'started', 'stopped', 'running', 'exception', 'starting', 'stopping']
        self.machine = Machine(model=self, states=states, initial='none')

    @staticmethod
    def help():
        return {}

    def run(self):
        raise NotImplementedError

    def running(self):
        return not self.event.is_set()

    def start(self):
        self.thread.start()

    def stop(self):
        self.event.set()


class ServiceState(enum.Enum):
    running = enum.auto()
    finished = enum.auto()
    stopped = enum.auto()


# create from json and to json methods
@dataclasses.dataclass
class ServiceInfo:
    name: str
    state: ServiceState
    service: BaseService


def on_begin(name, *args, **kwargs):
    ...


def on_error(name, *args, **kwargs):
    # if the service has recovery option, run it
    # else end
    ...


def on_end(name, *args, **kwargs):
    ...


def create(api):
    name = api.input('name')
    priority = api.input('priority')
    mode = api.input('mode')
    script = api.input('script')
    kwargs = {}
    for k in get_class(script).params():
        kwargs[k] = api.input(k)

    # on error
    # on recovery
    create_args = {
        'name': name,
        'priority': priority,
        'mode': mode,
        'script': script,
        'status': True,
        'kwargs': kwargs
    }

    settings = json.loads(open('iva/iva.json', 'r').read())
    services = settings.get('services', [])

    services.append(create_args)

    settings['services'] = services
    open('iva/iva.json', 'w').write(json.dumps(settings, indent=4, cls=JSONEncoder))


def get_class(name):
    service = None

    settings = json.loads(open('iva/iva.json', 'r').read())
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
    settings = json.loads(open('iva/iva.json', 'r').read())
    task_info = first(filter(lambda x: x['status'] and x['name'] == name, settings.get('services', [])))

    if task_info is None:
        task_info = {'script': name, 'kwargs': {}}

    script = task_info['script']

    service = get_class(script)

    if service is None:
        api.output('service is not found')
        return None

    kwargs = {**task_info['kwargs'], 'parent_os_path': api.parent_os_path, 'os_path': api.os_path}

    service = service(api, **kwargs)
    return service
