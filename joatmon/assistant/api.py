import datetime
import json
import os
import sys
import threading
import time

from joatmon.assistant import (
    service,
    task
)
from joatmon.assistant.service import (
    ServiceInfo,
    ServiceState
)
from joatmon.assistant.task import (
    TaskInfo,
    TaskState
)
from joatmon.system.lock import RWLock


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

    def __init__(self):
        settings = json.loads(open(os.path.join(os.environ.get('ASSISTANT_HOME'), 'system.json'), 'r').read())

        self.lock = RWLock()
        self.running_tasks = {}  # running, finished
        self.running_services = {}  # running, enabled, disabled, stopped, finished

        self.event = threading.Event()

        tasks = settings.get('tasks', [])
        for _task in sorted(filter(lambda x: x['status'] and x['on'] == 'startup', tasks), key=lambda x: x['priority']):
            self.run_task(_task['name'])  # need to do them in background

        services = settings.get('services', [])
        for _service in sorted(
                filter(lambda x: x['status'] and x['mode'] == 'automatic', services), key=lambda x: x['priority']
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
            settings = json.loads(open(os.path.join(os.environ.get('ASSISTANT_HOME'), 'system.json'), 'r').read())
            tasks = settings.get('tasks', [])
            tasks = filter(lambda x: x['status'], tasks)
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
                self.run_task(task_name=_task['name'], kwargs=None, background=True)  # need to do them in background
            for _task in sorted(old_tasks, key=lambda x: x['priority']):
                self.run_task(task_name=_task['name'], kwargs=None, background=True)  # need to do them in background

            time.sleep(1)

    def run_services(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        settings = json.loads(open(os.path.join(os.environ.get('ASSISTANT_HOME'), 'system.json'), 'r').read())

        services = settings.get('services', [])

        # if the service is closed for some reason and is configured as restart automatically, need to restart the service

        while not self.event.is_set():
            for _service in sorted(filter(lambda x: x['status'], services), key=lambda x: x['priority']):
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

    def do_command(self, command, action=None, arguments=None):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        try:
            if command is None or command == '':
                return False

            match command.lower():
                case 'list processes':
                    for k, v in self.running_tasks.items():
                        print(f'{v.name}: {v.state}')
                    for k, v in self.running_services.items():
                        print(f'{v.name}: {v.state}')

                    return False
                case 'exit':
                    return self.exit()
                case 'activate':
                    return self.run_task(task_name=action, kwargs=arguments)
                case _:
                    raise ValueError(f'wanted command is {command}, default case is not implemented')
        except Exception as ex:
            print(str(ex))  # use stacktrace and write all exception details, line number, function name, file name etc.

    def run_task(self, task_name, kwargs=None, background=False):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        _task = task.get(self, task_name, kwargs, background)
        if _task is None:
            return False

        if _task.background:
            if task_name not in self.running_tasks:
                self.running_tasks[task_name] = TaskInfo(task_name, TaskState.running, _task)
            else:
                self.running_tasks[task_name].state = TaskState.running
                self.running_tasks[task_name].task = _task
        _task.start()

        return False

    def start_service(self, service_name):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        _task = service.get(self, service_name)
        if _task is None:
            return False

        if service_name not in self.running_services:
            self.running_services[service_name] = ServiceInfo(service_name, ServiceState.running, _task)
        else:
            self.running_services[service_name].state = ServiceState.running
            self.running_services[service_name].service = _task
        _task.start()

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

        settings = json.loads(open(os.path.join(os.environ.get('ASSISTANT_HOME'), 'system.json'), 'r').read())

        tasks = settings.get('tasks', [])
        for _task in sorted(
                filter(lambda x: x['status'] and x['on'] == 'shutdown', tasks), key=lambda x: x['priority']
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
        return True

    def mainloop(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        while not self.event.is_set():
            time.sleep(0.1)
        self.exit()


if sys.platform == 'win32':
    class COSMO:
        def __init__(self):
            print('COSMO is supported on Windows')
            self.api = API()

        def action(self, command, action, arguments):
            self.api.do_command(command, action, arguments)

        def run(self):
            self.api.mainloop()
elif sys.platform == 'darwin':
    class COSMO:
        def __init__(self):
            print('COSMO is supported on MAC')
            self.api = API()

        def action(self, command, action, arguments):
            self.api.do_command(command, action, arguments)

        def run(self):
            self.api.mainloop()
elif sys.platform == 'linux':
    class COSMO:
        def __init__(self):
            print('COSMO is supported on Linux')
            self.api = API()

        def action(self, command, action, arguments):
            self.api.do_command(command, action, arguments)

        def run(self):
            self.api.mainloop()
else:
    raise ValueError(f'unsupported platform: {sys.platform}')
