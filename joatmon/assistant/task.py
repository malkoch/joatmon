import dataclasses
import datetime
import enum
import functools
import importlib.util
import json
import os
import threading

from transitions import Machine

from joatmon.event import Event
from joatmon.utility import (
    first,
    get_module_classes,
    JSONEncoder
)


class BaseTask:
    def __init__(self, api, background=False, **kwargs):
        self.api = api
        self.kwargs = kwargs
        self._background = background
        if self.background:
            self.thread = threading.Thread(target=self.run)
        self.stop_event = threading.Event()

        self.events = {
            'begin': Event(),
            'end': Event(),
            'error': Event()
        }

        states = ['none', 'started', 'stopped', 'running', 'exception', 'starting', 'stopping']
        self.machine = Machine(model=self, states=states, initial='none')

    @property
    def background(self):
        return self._background

    @background.setter
    def background(self, value):
        self._background = value
        if self.background:
            self.thread = threading.Thread(target=self.run)

    @staticmethod
    def help():
        return {}

    def run(self):
        raise NotImplementedError

    def running(self):
        return not self.stop_event.is_set()

    def start(self):
        self.events['begin'].fire()

        if self.background:
            self.thread.start()
        else:
            self.run()

    def stop(self):
        self.stop_event.set()

        self.events['end'].fire()


class TaskState(enum.Enum):
    running = enum.auto()
    finished = enum.auto()


# create from json and to json methods
@dataclasses.dataclass
class TaskInfo:
    name: str
    state: TaskState
    task: BaseTask


def on_begin(name, *args, **kwargs):
    settings = json.loads(open('iva.json', 'r').read())
    tasks = settings.get('tasks', [])

    task_info = first(filter(lambda x: x['name'] == name, tasks))
    if task_info is None:
        return

    task_info['last_run_time'] = datetime.datetime.now()
    if task_info['on'] == 'interval':
        task_info['next_run_time'] = task_info['last_run_time'] + datetime.timedelta(seconds=task_info['interval'])

    for idx, task in enumerate(tasks):
        if task['name'] == task_info['name']:
            tasks[idx] = task_info
            break

    settings['tasks'] = tasks
    open('iva.json', 'w').write(json.dumps(settings, indent=4, cls=JSONEncoder))


def on_error(name, *args, **kwargs):
    ...


def on_end(name, *args, **kwargs):
    ...


def create(api):
    name = api.listen('name')
    priority = api.listen('priority')
    on = api.listen('on')
    script = api.listen('script')
    kwargs = {}
    for k in get_class(script).params():
        kwargs[k] = api.listen(k)

    # need last run time
    # need next run time
    # need last run result
    # need interval
    # if on == 'interval' need to as for interval as well
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
    open('iva.json', 'w').write(json.dumps(settings, indent=4, cls=JSONEncoder))


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


def get(api, name, kwargs, background):
    settings = json.loads(open('iva.json', 'r').read())
    task_info = first(filter(lambda x: x['status'] and x['name'] == name, settings.get('tasks', [])))

    if task_info is None:
        task_info = {'script': name, 'kwargs': {}}

    script = task_info['script']

    task = get_class(script)

    if task is None:
        api.say(f'task {name} is not found')
        return None

    kwargs = {**(kwargs or {}), **task_info['kwargs'], 'parent_os_path': api.parent_os_path, 'os_path': api.os_path}

    task = task(api, **kwargs)
    task.background = background

    task.events['begin'] += functools.partial(on_begin, name=name)
    task.events['end'] += functools.partial(on_end, name=name)
    task.events['error'] += functools.partial(on_error, name=name)
    return task
