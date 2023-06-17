import argparse
import dataclasses
import enum
import importlib.util
import inspect
import json
import os
import threading
import time

from joatmon import context
from joatmon.assistant.intents import GenericAssistant
from joatmon.assistant.service import BaseService
from joatmon.assistant.task import BaseTask
from joatmon.hid.microphone import InputDriver
from joatmon.hid.speaker import OutputDevice
from joatmon.system.lock import RWLock
from joatmon.utility import (
    first,
    get_module_classes
)


class CTX:
    ...


ctx = CTX()
context.set_ctx(ctx)


def _get_task(script_name):
    task = None

    settings = json.loads(open('iva.json', 'r').read())
    for scripts in settings.get('scripts', []):
        if os.path.isabs(scripts):
            if os.path.exists(scripts) and os.path.exists(os.path.join(scripts, f'{script_name}.py')):
                spec = importlib.util.spec_from_file_location(scripts, os.path.join(scripts, f'{script_name}.py'))
                action_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(action_module)
            else:
                continue
        else:
            try:
                _module = __import__(scripts, fromlist=[f'{script_name}'])
            except ModuleNotFoundError:
                continue

            action_module = getattr(_module, script_name, None)

        if action_module is None:
            continue

        for class_ in get_module_classes(action_module):
            if not issubclass(class_[1], BaseTask) or class_[1] is BaseTask:
                continue

            task = class_[1]

    return task


def _get_service(script_name):
    task = None

    settings = json.loads(open('iva.json', 'r').read())
    for scripts in settings.get('scripts', []):
        if os.path.isabs(scripts):
            if os.path.exists(scripts) and os.path.exists(os.path.join(scripts, f'{script_name}.py')):
                spec = importlib.util.spec_from_file_location(scripts, os.path.join(scripts, f'{script_name}.py'))
                action_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(action_module)
            else:
                continue
        else:
            try:
                _module = __import__(scripts, fromlist=[f'{script_name}'])
            except ModuleNotFoundError:
                continue

            action_module = getattr(_module, script_name, None)

        if action_module is None:
            continue

        task = getattr(action_module, 'Service', None)

    return task


def create_task(task):
    create_args = {
        'name': task.pop('name', ''),
        'priority': int(task.pop('priority', '')),
        'on': task.pop('on', ''),
        'script': task.pop('script', ''),
        'status': True,
        'position': task.pop('position', ''),
        'args': task
    }

    settings = json.loads(open('iva.json', 'r').read())
    tasks = settings.get('tasks', [])

    tasks.append(create_args)

    settings['tasks'] = tasks
    open('iva.json', 'w').write(json.dumps(settings, indent=4))


def create_service(service):
    create_args = {
        'name': service.pop('name', ''),
        'priority': int(service.pop('priority', '')),
        'mode': service.pop('mode', ''),
        'script': service.pop('script', ''),
        'status': True,
        'position': service.pop('position', ''),
        'args': service
    }

    settings = json.loads(open('iva.json', 'r').read())
    tasks = settings.get('services', [])

    tasks.append(create_args)

    settings['services'] = tasks
    open('iva.json', 'w').write(json.dumps(settings, indent=4))


class TaskState(enum.Enum):
    running = enum.auto()
    finished = enum.auto()


@dataclasses.dataclass
class TaskInfo:
    name: str
    state: TaskState
    task: BaseTask


class ServiceState(enum.Enum):
    running = enum.auto()
    finished = enum.auto()
    stopped = enum.auto()


@dataclasses.dataclass
class ServiceInfo:
    name: str
    state: ServiceState
    service: BaseService


class Interpreter:
    def __init__(self):
        settings = json.loads(open('iva.json', 'r').read())

        self.assistant = GenericAssistant('iva.json', model_name="iva")
        self.assistant.train_model()
        self.assistant.save_model('iva')

        self.parent_os_path = os.path.abspath(os.path.curdir)
        self.os_path = os.sep

        self.output_device = OutputDevice()
        self.input_device = InputDriver(self.output_device)

        self.lock = RWLock()
        self.running_tasks = {}  # running, finished
        self.running_services = {}  # running, enabled, disabled, stopped, finished

        self.event = threading.Event()

        tasks = settings.get('tasks', [])
        for task in sorted(filter(lambda x: x['status'] and x['on'] == 'startup', tasks), key=lambda x: x['priority']):
            self.run_task(task['name'])  # need to do them in background
        services = settings.get('services', [])
        for service in sorted(filter(lambda x: x['status'] and x['mode'] == 'automatic', services), key=lambda x: x['priority']):
            self.start_service(service['name'])  # need to do them in background

        self.cleaning_thread = threading.Thread(target=self.clean)
        self.cleaning_thread.start()
        self.service_thread = threading.Thread(target=self.run_services)
        self.service_thread.start()

        # self.do_action('ls .')
        # self.do_action('dt')

    def run_services(self):
        settings = json.loads(open('iva.json', 'r').read())

        services = settings.get('services', [])

        # if the service is closed for some reason and it is configured as restart automatically, need to restart the service

        while not self.event.is_set():
            for service in sorted(filter(lambda x: x['status'], services), key=lambda x: x['priority']):
                if service['name'] not in self.running_services or self.running_services[service['name']].state == ServiceState.finished:
                    self.start_service(service['name'])  # need to do them in background
            time.sleep(1)

    def clean(self):
        while not self.event.is_set():
            with self.lock.r_locked():
                task_keys = [key for key in self.running_tasks.keys()]
                service_keys = [key for key in self.running_services.keys()]

            delete_task_keys = []
            for key in task_keys:
                task_info = self.running_tasks[key]
                if not task_info.task.running():
                    delete_task_keys.append(key)
            delete_service_keys = []
            for key in service_keys:
                task_info = self.running_services[key]
                if not task_info.service.running() and task_info.state != ServiceState.stopped:
                    delete_service_keys.append(key)

            for key in delete_task_keys:
                with self.lock.w_locked():
                    del self.running_tasks[key]
            for key in delete_service_keys:
                with self.lock.w_locked():
                    del self.running_services[key]

            time.sleep(1)

    def listen(self, prompt=None):
        response = self.input_device.listen(prompt)
        intent, prob = self.intent(response)
        if prob > .9:
            return intent
        else:
            return response

    def say(self, text):
        self.output_device.say(text)

    def intent(self, text):
        return self.assistant.request(text)

    def do_action(self, line):
        try:
            if line is None or line == '':
                return False

            match line.lower():
                case 'enable':
                    return self.enable()
                case 'disable':
                    return self.disable()
                case 'update':
                    return self.update()
                case 'delete':
                    return self.delete()
                case 'configure':
                    return self.configure()
                case 'run':
                    return self.run()
                case 'start':
                    return self.start()
                case 'stop':
                    return self.stop()
                case 'restart':
                    return self.restart()
                case 'skip':
                    return False
                case 'help':
                    return self.help()
                case 'exit':
                    return self.exit()
                case _:
                    return self.run_task(line)
        except Exception as ex:
            print(str(ex))  # use stacktrace and write all exception details, line number, function name, file name etc.
            # return self.exit()

    def enable(self, *args):
        for arg in args:
            ...

        settings = json.loads(open('iva.json', 'r').read())
        open('iva.json', 'w').write(json.dumps(settings, indent=4))

        return False

    def disable(self, *args):
        for arg in args:
            ...

        settings = json.loads(open('iva.json', 'r').read())
        open('iva.json', 'w').write(json.dumps(settings, indent=4))

        return False

    def update(self):
        ...

    def delete(self):
        ...

    def configure(self):
        ...

    def run(self):
        self.say('what do you want me to run')
        action_type = self.listen()

        if action_type.lower() == 'task':
            self.say('which task do you want me to run')
            task = self.listen()
            return self.run_task(task)

        return False

    def run_task(self, task_name, kwargs=None):
        settings = json.loads(open('iva.json', 'r').read())
        task_info = first(filter(lambda x: x['name'] == task_name, settings.get('tasks', [])))

        if task_info is None:
            task_info = {'script': task_name, 'args': {}}

        script = task_info['script']

        task = _get_task(script)

        if task is None:
            self.say(f'task {task_name} is not found')
            return False

        kwargs = {**(kwargs or {}), **task_info['args'], 'parent_os_path': self.parent_os_path, 'os_path': self.os_path}

        task = task(self, **kwargs)
        if task_name not in self.running_tasks:
            self.running_tasks[task_name] = TaskInfo(task_name, TaskState.running, task)
        else:
            self.running_tasks[task_name].state = TaskState.running
            self.running_tasks[task_name].task = task
        task.start()

        return False

    def start(self):
        self.say('what service do you want me to start')
        action = self.listen()
        return self.start_service(action)

    def start_service(self, service_name):
        settings = json.loads(open('iva.json', 'r').read())
        task_info = first(filter(lambda x: x['name'] == service_name, settings.get('services', [])))

        if task_info is None:
            task_info = {'script': service_name, 'args': {}}
            # return False

        script = task_info['script']

        task = _get_service(script)

        if task is None:
            self.say('service is not found')
            return False

        args = task_info['args']
        args['parent_os_path'] = self.parent_os_path
        args['os_path'] = self.os_path

        task = task(self, *[], **args)
        if service_name not in self.running_services:
            self.running_services[service_name] = ServiceInfo(service_name, ServiceState.running, task)
        else:
            self.running_services[service_name].state = ServiceState.running
            self.running_services[service_name].service = task
        task.start()

        return False

    def stop(self):
        self.say('what service do you want me to stop')
        action = self.listen()
        return self.stop_service(action)

    def stop_service(self, service_name):
        self.running_services[service_name].state = ServiceState.stopped
        self.running_services[service_name].service.stop()
        return False

    def restart(self):
        self.say('what service do you want me to restart')
        action = self.listen()
        return self.restart_service(action)

    def restart_service(self, service_name):
        self.stop_service(service_name)
        self.start_service(service_name)
        return False

    def config(self, *args):
        parser = argparse.ArgumentParser()
        parser.add_argument('--create', type=str)
        parser.add_argument('--update', type=str)
        parser.add_argument('--delete', type=str)
        parser.add_argument('--value', type=str)

        namespace, _ = parser.parse_known_args(args)

        if namespace.create:
            config = {
                'action': 'create',
                'name': namespace.create,
                'value': namespace.value
            }
        elif namespace.update:
            config = {
                'action': 'update',
                'name': namespace.update,
                'value': namespace.value
            }
        elif namespace.delete:
            config = {
                'action': 'delete',
                'name': namespace.delete
            }
        else:
            raise ValueError('')

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

    def help(self):
        settings = json.loads(open('iva.json', 'r').read())
        for scripts in settings.get('scripts', []):
            if os.path.isabs(scripts) and os.path.exists(scripts):
                for module in list(filter(lambda x: '__' not in x, map(lambda x: x.replace('.py', ''), os.listdir(scripts.replace('.', '/'))))):
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

        return False

    def exit(self):
        settings = json.loads(open('iva.json', 'r').read())

        tasks = settings.get('tasks', [])
        for task in sorted(filter(lambda x: x['status'] and x['on'] == 'shutdown', tasks), key=lambda x: x['priority']):
            self.run_task(task['name'])

        with self.lock.r_locked():
            task_keys = [key for key in self.running_tasks.keys()]
            service_keys = [key for key in self.running_services.keys()]

        for key in task_keys:
            task_info = self.running_tasks[key]
            task_info.task.stop()
        for key in service_keys:
            self.stop_service(key)

        self.event.set()
        self.input_device.stop()
        return True

    def mainloop(self):
        while not self.event.is_set():
            command = self.listen()
            self.do_action(command)


def main():
    interpreter = Interpreter()
    interpreter.mainloop()
