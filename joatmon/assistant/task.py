import dataclasses
import enum
import importlib.util
import json
import os
import threading

from joatmon.utility import (
    first,
    get_module_classes
)


class BaseTask:
    def __init__(self, api, background=False, **kwargs):
        self.api = api
        self.kwargs = kwargs
        self.background = background
        if self.background:
            self.thread = threading.Thread(target=self.run)
        self.event = threading.Event()

    @staticmethod
    def params():
        return ['todo']

    def run(self):
        raise NotImplementedError

    def running(self):
        return not self.event.is_set()

    def start(self):
        if self.background:
            self.thread.start()
        else:
            self.run()

    def stop(self):
        self.event.set()


class TaskState(enum.Enum):
    running = enum.auto()
    finished = enum.auto()


@dataclasses.dataclass
class TaskInfo:
    name: str
    state: TaskState
    task: BaseTask


def create(api):
    name = api.listen('name')
    priority = api.listen('priority')
    on = api.listen('on')
    script = api.listen('script')
    kwargs = {}
    for k in get_class(script).params():
        kwargs[k] = api.listen(k)

    create_args = {
        'name': name,
        'priority': priority,
        'on': on,
        'script': script,
        'status': True,
        'kwargs': kwargs
    }

    settings = json.loads(open('iva.json', 'r').read())
    tasks = settings.get('tasks', [])

    tasks.append(create_args)

    settings['tasks'] = tasks
    open('iva.json', 'w').write(json.dumps(settings, indent=4))


def get_class(name):
    task = None

    settings = json.loads(open('iva.json', 'r').read())
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

        for class_ in get_module_classes(action_module):
            if not issubclass(class_[1], BaseTask) or class_[1] is BaseTask:
                continue

            task = class_[1]

    return task


def get(api, name, kwargs):
    settings = json.loads(open('iva.json', 'r').read())
    task_info = first(filter(lambda x: x['name'] == name, settings.get('tasks', [])))

    if task_info is None:
        task_info = {'script': name, 'kwargs': {}}

    script = task_info['script']

    task = get_class(script)

    if task is None:
        api.say(f'task {name} is not found')
        return None

    kwargs = {**(kwargs or {}), **task_info['kwargs'], 'parent_os_path': api.parent_os_path, 'os_path': api.os_path}

    task = task(api, **kwargs)
    return task
