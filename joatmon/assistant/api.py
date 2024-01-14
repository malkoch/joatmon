import datetime
import importlib.util
import json
import os
import threading
import time

from joatmon.assistant.service import (BaseService, ServiceInfo, ServiceState)
from joatmon.assistant.task import (
    BaseTask, TaskInfo,
    TaskState
)
from joatmon.core.utility import first, get_module_classes
from joatmon.system.lock import RWLock


# instead of reading from system.json, get these values as an argument
# after creating task or service, need to restart the system
class API:
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

    def __init__(self, base_folder, action_folders, tasks=None, services=None):
        self.cwd = base_folder
        self.folders = action_folders

        self.tasks = tasks or []
        self.services = services or []

        self.lock = RWLock()
        self.running_tasks = {}  # running, finished
        self.running_services = {}  # running, enabled, disabled, stopped, finished

        self.event = threading.Event()

        for _task in sorted(filter(lambda x: x['status'] and x['on'] == 'startup', self.tasks), key=lambda x: x['priority']):
            self.run_task(_task['name'])  # need to do them in background

        for _service in sorted(
                filter(lambda x: x['status'] and x['mode'] == 'automatic', self.services), key=lambda x: x['priority']
        ):
            self.start_service(_service['name'])  # need to do them in background

        self.cleaning_thread = threading.Thread(target=self.clean)
        self.cleaning_thread.start()
        time.sleep(1)

        self.service_thread = threading.Thread(target=self.run_services)
        self.service_thread.start()
        time.sleep(1)

        self.interval_thread = threading.Thread(target=self.run_interval)
        self.interval_thread.start()
        time.sleep(1)

    def run_interval(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        while not self.event.is_set():
            tasks = filter(lambda x: x['status'], self.tasks)
            tasks = filter(lambda x: x['on'] == 'interval', tasks)
            tasks = filter(lambda x: x['interval'] > 0, tasks)
            tasks = list(tasks)

            new_tasks = filter(lambda x: x.get('last_run_time', None) is None, tasks)
            new_tasks = list(new_tasks)
            old_tasks = filter(lambda x: x.get('next_run_time', None) is not None, tasks)
            old_tasks = filter(
                lambda x: datetime.datetime.now() > datetime.datetime.fromisoformat(x['next_run_time']), old_tasks
            )
            old_tasks = list(old_tasks)

            for _task in sorted(new_tasks, key=lambda x: x['priority']):
                self.run_task(task_name=_task['name'], kwargs=None)  # need to do them in background
            for _task in sorted(old_tasks, key=lambda x: x['priority']):
                self.run_task(task_name=_task['name'], kwargs=None)  # need to do them in background

            time.sleep(1)

    def run_services(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        # if the service is closed for some reason and is configured as restart automatically, need to restart the service

        while not self.event.is_set():
            for _service in sorted(filter(lambda x: x['status'], self.services), key=lambda x: x['priority']):
                if (
                        _service['name'] not in self.running_services
                        or self.running_services[_service['name']].state == ServiceState.finished
                ):
                    self.start_service(_service['name'])  # need to do them in background
            time.sleep(1)

    def clean(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        # maybe call stop function of the task or service
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
                    self.running_tasks[key].task.stop()
                    del self.running_tasks[key]
            for key in delete_service_keys:
                with self.lock.w_locked():
                    self.running_services[key].service.stop()
                    del self.running_services[key]

            time.sleep(1)

    def action(self, action, arguments):  # each request must have request id and client id or token
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        try:
            if action is None or action == '':
                return False

            match action.lower():
                case 'list processes':
                    for k, v in self.running_tasks.items():
                        print(f'{v.name}: {v.state}')
                    for k, v in self.running_services.items():
                        print(f'{v.name}: {v.state}')

                    return False
                case 'exit':
                    return self.exit()
                case _:
                    return self.run_task(task_name=action, kwargs=arguments)
        except Exception as ex:
            print(str(ex))  # use stacktrace and write all exception details, line number, function name, file name etc.

    def run_task(self, task_name, kwargs=None):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        settings = json.loads(open(os.path.join(os.environ.get('ASSISTANT_HOME'), 'system.json'), 'r').read())
        task_info = first(filter(lambda x: x['status'] and x['name'] == task_name, settings.get('tasks', [])))

        if task_info is None:
            task_info = {'script': task_name, 'kwargs': {}}

        script = task_info['script']

        task = None

        settings = json.loads(open(os.path.join(os.environ.get('ASSISTANT_HOME'), 'system.json'), 'r').read())
        for scripts in settings.get('scripts', []):
            if os.path.isabs(scripts):
                if os.path.exists(scripts) and os.path.exists(os.path.join(scripts, f'{script}.py')):
                    spec = importlib.util.spec_from_file_location(scripts, os.path.join(scripts, f'{script}.py'))
                    action_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(action_module)
                else:
                    continue
            else:
                try:
                    _module = __import__(scripts, fromlist=[f'{script}'])
                except ModuleNotFoundError as ex:
                    print(str(ex))
                    continue

                action_module = getattr(_module, script, None)

            if action_module is None:
                continue

            for class_ in get_module_classes(action_module):
                if not issubclass(class_[1], BaseTask) or class_[1] is BaseTask:
                    continue

                task = class_[1]

        if task is None:
            return False

        kwargs = {**(kwargs or {}), **task_info.get('kwargs', {})}

        task = task(task_name, self, **kwargs)

        if task_name not in self.running_tasks:
            self.running_tasks[task_name] = TaskInfo(task_name, TaskState.running, task)
        else:
            self.running_tasks[task_name].state = TaskState.running
            self.running_tasks[task_name].task = task
        task.start()

        return False

    def start_service(self, service_name):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        task_info = first(filter(lambda x: x['status'] and x['name'] == service_name, self.services))

        if task_info is None:
            task_info = {'script': service_name, 'kwargs': {}}

        script = task_info['script']

        service = None

        settings = json.loads(open(os.path.join(os.environ.get('ASSISTANT_HOME'), 'system.json'), 'r').read())
        for scripts in settings.get('scripts', []):
            if os.path.isabs(scripts):
                if os.path.exists(scripts) and os.path.exists(os.path.join(scripts, f'{script}.py')):
                    spec = importlib.util.spec_from_file_location(scripts, os.path.join(scripts, f'{script}.py'))
                    action_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(action_module)
                else:
                    continue
            else:
                try:
                    _module = __import__(scripts, fromlist=[f'{script}'])
                except ModuleNotFoundError:
                    continue

                action_module = getattr(_module, script, None)

            if action_module is None:
                continue

            for class_ in get_module_classes(action_module):
                if not issubclass(class_[1], BaseService) or class_[1] is BaseService:
                    continue

                service = class_[1]

        if service is None:
            return False

        kwargs = task_info.get('kwargs', {})

        service = service(service_name, self, **kwargs)

        if service_name not in self.running_services:
            self.running_services[service_name] = ServiceInfo(service_name, ServiceState.running, service)
        else:
            self.running_services[service_name].state = ServiceState.running
            self.running_services[service_name].service = service
        service.start()

        return False

    def stop_service(self, service_name):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.running_services[service_name].state = ServiceState.stopped
        self.running_services[service_name].service.stop()
        return False

    def restart_service(self, service_name):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.stop_service(service_name)
        self.start_service(service_name)
        return False

    def exit(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        for _task in sorted(
                filter(lambda x: x['status'] and x['on'] == 'shutdown', self.tasks), key=lambda x: x['priority']
        ):
            self.run_task(_task['name'])

        with self.lock.r_locked():
            task_keys = [key for key in self.running_tasks.keys()]
            service_keys = [key for key in self.running_services.keys()]

        for key in task_keys:
            task_info = self.running_tasks[key]
            task_info.task.stop()
        for key in service_keys:
            self.stop_service(key)

        self.event.set()

        exit(1)
