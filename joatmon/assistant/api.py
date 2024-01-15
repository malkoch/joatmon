import asyncio
import datetime
import importlib.util
import os
import typing
import uuid

from joatmon.assistant.runnable import Runnable
from joatmon.assistant.service import (BaseService, Service)
from joatmon.assistant.task import (
    BaseTask, Task
)
from joatmon.core.utility import first, get_module_classes


# instead of reading from system.json, get these values as an argument
# after creating task or service, need to restart the system
# make this async, we can get exceptions from it as well
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

    def __init__(self, loop, cwd, folders=None, tasks: typing.Optional[typing.List[Task]] = None, services: typing.Optional[typing.List[Service]] = None):
        self.loop = loop
        self.cwd = cwd
        self.folders = folders or []

        self.tasks = tasks or []
        self.services = services or []

        self.processes: typing.List[Runnable] = []

        self.event = asyncio.Event()

    async def main(self):
        for _task in sorted(filter(lambda x: x.status and x.on == 'startup', self.tasks), key=lambda x: x.priority):
            self.run_task(_task.name)

        for _service in sorted(filter(lambda x: x.status and x.mode == 'automatic', self.services), key=lambda x: x.priority):
            self.start_service(_service.name)

        self.cleaning_task = asyncio.ensure_future(self.clean(), loop=self.loop)
        self.service_task = asyncio.ensure_future(self.run_services(), loop=self.loop)
        self.interval_task = asyncio.ensure_future(self.run_interval(), loop=self.loop)

        while not self.event.is_set():
            try:
                done, pending = await asyncio.wait(
                    list(map(lambda x: x.task, self.processes)) + [self.cleaning_task, self.service_task, self.interval_task],
                    return_when=asyncio.FIRST_COMPLETED
                )

                if self.cleaning_task in done:
                    self.cleaning_task = asyncio.ensure_future(self.clean(), loop=self.loop)
                if self.service_task in done:
                    self.service_task = asyncio.ensure_future(self.run_services(), loop=self.loop)
                if self.interval_task in done:
                    self.interval_task = asyncio.ensure_future(self.run_interval(), loop=self.loop)
            except Exception as ex:
                print(str(ex))

            await asyncio.sleep(0.1)

    async def run_interval(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
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

    async def run_services(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        # if the service is closed for some reason and is configured as restart automatically, need to restart the service

        for _service in sorted(filter(lambda x: x['status'], self.services), key=lambda x: x['priority']):
            if _service.id not in list(map(lambda x: x.service.id, self.processes)):
                self.start_service(_service['name'])  # need to do them in background

    async def clean(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ended_processes = list(filter(lambda x: not x.running(), self.processes))
        self.processes = list(filter(lambda x: x.running(), self.processes))

        for process in ended_processes:
            process.stop()

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
                    for process in self.processes:
                        if process.type == 'task':
                            print(f'{process.info.name}, {process.process_id}: {process.running()}')
                        if process.type == 'service':
                            print(f'{process.info.name}, {process.process_id}: {process.running()}')

                    return False
                case 'exit':
                    return self.exit()
                case _:
                    return self.run_task(task_name=action, kwargs=arguments)
        except Exception as ex:
            print(str(ex))  # use stacktrace and write all exception details, line number, function name, file name etc.

    def _run_task(self, task: Task, **kwargs):
        cls = None
        for scripts in self.folders:
            if os.path.isabs(scripts):
                if os.path.exists(scripts) and os.path.exists(os.path.join(scripts, f'{task.script}.py')):
                    spec = importlib.util.spec_from_file_location(scripts, os.path.join(scripts, f'{task.script}.py'))
                    action_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(action_module)
                else:
                    continue
            else:
                try:
                    _module = __import__(scripts, fromlist=[f'{task.script}'])
                except ModuleNotFoundError as ex:
                    print(str(ex))
                    continue

                action_module = getattr(_module, task.script, None)

            if action_module is None:
                continue

            for class_ in get_module_classes(action_module):
                if not issubclass(class_[1], Runnable):
                    continue

                if class_[1] is Runnable:
                    continue

                if not issubclass(class_[1], BaseTask):
                    continue

                if class_[1] is BaseTask:
                    continue

                cls = class_[1]

        if cls is None:
            return False

        kwargs = {**(kwargs or {}), **task.arguments}

        obj = cls(task, self, **kwargs)
        obj.start()

        self.processes.append(obj)

    def run_task(self, task_name, kwargs=None):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        task_info: typing.Optional[Task] = first(filter(lambda x: x.status and x.name == task_name, self.tasks))
        if task_info is None:
            task_info = Task(
                id=str(uuid.uuid4()),
                name=task_name,
                description='',
                script=task_name,
                status=True,
                on='manual',
                priority=0,
                arguments={},
                created_at=datetime.datetime.now(),
                updated_at=datetime.datetime.now(),
            )
        self._run_task(task_info, **kwargs)

        return False

    def _start_service(self, service: Service):
        service_cls = None

        for scripts in self.folders:
            if os.path.isabs(scripts):
                if os.path.exists(scripts) and os.path.exists(os.path.join(scripts, f'{service.script}.py')):
                    spec = importlib.util.spec_from_file_location(scripts, os.path.join(scripts, f'{service.script}.py'))
                    action_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(action_module)
                else:
                    continue
            else:
                try:
                    _module = __import__(scripts, fromlist=[f'{service.script}'])
                except ModuleNotFoundError:
                    continue

                action_module = getattr(_module, service.script, None)

            if action_module is None:
                continue

            for class_ in get_module_classes(action_module):
                if not issubclass(class_[1], BaseService) or class_[1] is BaseService:
                    continue

                service_cls = class_[1]

        if service_cls is None:
            return False

        kwargs = service_cls.arguments

        obj = service_cls(service, self, **kwargs)
        obj.start()

        self.processes.append(obj)

    def start_service(self, service_name):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        task_info = first(filter(lambda x: x.status and x.name == service_name, self.services))

        if task_info is None:
            return False

        self._start_service(task_info)

    def stop_service(self, service_name):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """

        for process in self.processes:
            if process.type == 'service' and process.info.name == service_name:
                process.stop()
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

        for process in self.processes:
            process.stop()

        self.event.set()

        exit(1)
