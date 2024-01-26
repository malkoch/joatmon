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
from joatmon.core.event import Event


class API:
    """
    API class for managing tasks and services.

    This class provides a way to manage tasks and services, including running tasks, starting and stopping services, and cleaning up processes.

    Attributes:
        loop (asyncio.AbstractEventLoop): The event loop where the tasks and services are run.
        cwd (str): The current working directory.
        folders (list): The folders where the scripts for tasks and services are located.
        tasks (list): The list of tasks.
        services (list): The list of services.
        processes (list): The list of running processes.
        event (asyncio.Event): An event for signaling the termination of the API.

    Args:
        loop (asyncio.AbstractEventLoop): The event loop where the tasks and services are run.
        cwd (str): The current working directory.
        folders (list, optional): The folders where the scripts for tasks and services are located.
        tasks (list, optional): The list of tasks.
        services (list, optional): The list of services.
    """

    def __init__(
        self,
        loop,
        cwd,
        folders=None,
        tasks: typing.Optional[typing.List[Task]] = None,
        services: typing.Optional[typing.List[Service]] = None,
        events: typing.Optional[typing.Dict[str, Event]] = None
    ):
        """
        Initialize the API.

        Args:
            loop (asyncio.AbstractEventLoop): The event loop where the tasks and services are run.
            cwd (str): The current working directory.
            folders (list, optional): The folders where the scripts for tasks and services are located.
            tasks (list, optional): The list of tasks.
            services (list, optional): The list of services.
        """
        self.loop = loop
        self.cwd = cwd
        self.folders = folders or []

        self.tasks = tasks or []
        self.services = services or []

        self.processes: typing.List[Runnable] = []

        self.event = asyncio.Event()
        self.events = events

    async def main(self):
        """
        Main method for running the API.

        This method starts the tasks and services, and waits for them to complete. It also handles cleaning up the processes.
        """
        for _task in sorted(filter(lambda x: x.status and x.on == 'startup', self.tasks), key=lambda x: x.priority):
            self.run_task(_task.name)

        for _service in sorted(filter(lambda x: x.status and x.mode == 'automatic', self.services), key=lambda x: x.priority):
            self.start_service(_service.name)

        service_task = asyncio.ensure_future(self.run_services(), loop=self.loop)
        interval_task = asyncio.ensure_future(self.run_interval(), loop=self.loop)

        while not self.event.is_set():
            try:
                done, pending = await asyncio.wait(
                    list(filter(lambda a: a, map(lambda x: x.task, self.processes))) + [service_task, interval_task],
                    return_when=asyncio.FIRST_COMPLETED
                )

                for t in filter(lambda x: x not in (service_task, interval_task), done):
                    task = first(filter(lambda x: x.task == t, self.processes))
                    if task is not None:
                        if t.exception():
                            self.events.get('error', Event()).fire(info=task.info, ex=t.exception())
                        else:
                            self.events.get('end', Event()).fire(info=task.info)

                        task.info.last_run_time = datetime.datetime.now()
                        task.info.next_run_time = datetime.datetime.now() + datetime.timedelta(seconds=task.info.interval)

                self.processes = list(filter(lambda x: x.task and x.task not in done, self.processes))

                if service_task in done:
                    service_task = asyncio.ensure_future(self.run_services(), loop=self.loop)
                if interval_task in done:
                    interval_task = asyncio.ensure_future(self.run_interval(), loop=self.loop)
            except Exception as ex:
                print(str(ex))

            await asyncio.sleep(0.1)

    async def run_interval(self):
        """
        Run tasks at specified intervals.

        This method runs the tasks that are configured to run at specified intervals.
        """
        tasks = filter(lambda x: x.status, self.tasks)
        tasks = filter(lambda x: x.on == 'interval', tasks)
        tasks = filter(lambda x: x.interval > 0, tasks)
        tasks = list(tasks)

        new_tasks = filter(lambda x: x.last_run_time is None, tasks)
        new_tasks = list(new_tasks)
        old_tasks = filter(lambda x: x.next_run_time is not None, tasks)
        old_tasks = filter(lambda x: datetime.datetime.now() > x.next_run_time, old_tasks)
        old_tasks = list(old_tasks)

        for _task in sorted(new_tasks, key=lambda x: x.priority):
            self.run_task(task_name=_task.name, kwargs=None)  # need to do them in background
        for _task in sorted(old_tasks, key=lambda x: x.priority):
            self.run_task(task_name=_task.name, kwargs=None)  # need to do them in background

    async def run_services(self):
        """
        Run services.

        This method runs the services that are configured to run automatically.
        """
        # if the service is closed for some reason and is configured as restart automatically, need to restart the service

        for _service in sorted(filter(lambda x: x.status, self.services), key=lambda x: x.priority):
            if _service.id not in list(map(lambda x: x.info.id, self.processes)):
                self.start_service(_service.name)  # need to do them in background

    def action(self, action, arguments):  # each request must have request id and client id or token
        """
        Perform an action.

        This method performs an action based on the provided action name and arguments.

        Args:
            action (str): The name of the action to perform.
            arguments (dict): The arguments for the action.
        """
        try:
            if action is None or action == '':
                return False

            match action.lower():
                case 'list processes':
                    for process in self.processes:
                        print(f'{process.info.name}, {process.process_id}: {process.running()}')
                    return False
                case 'exit':
                    return self.exit()
                case _:
                    return self.run_task(task_name=action, kwargs=arguments)
        except Exception as ex:
            print(str(ex))  # use stacktrace and write all exception details, line number, function name, file name etc.

    def _run_task(self, task: Task, **kwargs):
        """
        Run a task.

        This method runs a task with the provided arguments.

        Args:
            task (Task): The task to run.
            kwargs (dict): The arguments for the task.
        """
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

        # check if the task is already running

        obj = cls(task, self, **kwargs)
        obj.start()

        self.events.get('begin', Event()).fire(info=obj.info)

        self.processes.append(obj)

    def run_task(self, task_name, kwargs=None):
        """
        Run a task by name.

        This method runs a task with the provided name and arguments.

        Args:
            task_name (str): The name of the task to run.
            kwargs (dict, optional): The arguments for the task.
        """
        kwargs = kwargs or {}

        task_info: typing.Optional[Task] = first(filter(lambda x: x.status and x.name == task_name, self.tasks))
        if task_info is None:
            task_info = Task(
                id=str(uuid.uuid4()),
                name=task_name,
                description='',
                script=task_name,
                status=True,
                on='manual',
                interval=999999999,
                priority=0,
                arguments={},
                created_at=datetime.datetime.now(),
                updated_at=datetime.datetime.now()
            )
        self._run_task(task_info, **kwargs)

        return False

    def _start_service(self, service: Service):
        """
        Start a service.

        This method starts a service.

        Args:
            service (Service): The service to start.
        """
        cls = None

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

                cls = class_[1]

        if cls is None:
            return False

        kwargs = service.arguments

        obj = cls(service, self, **kwargs)
        obj.start()

        self.events.get('begin', Event()).fire(info=obj.info)

        self.processes.append(obj)

    def start_service(self, service_name):
        """
        Start a service by name.

        This method starts a service with the provided name.

        Args:
            service_name (str): The name of the service to start.
        """
        task_info = first(filter(lambda x: x.status and x.name == service_name, self.services))

        if task_info is None:
            return False

        self._start_service(task_info)

    def stop_service(self, service_name):
        """
        Stop a service by name.

        This method stops a service with the provided name.

        Args:
            service_name (str): The name of the service to stop.
        """

        for process in self.processes:
            if process.type == 'service' and process.info.name == service_name:
                process.stop()
                return False

    def restart_service(self, service_name):
        """
        Restart a service by name.

        This method restarts a service with the provided name.

        Args:
            service_name (str): The name of the service to restart.
        """
        self.stop_service(service_name)
        self.start_service(service_name)
        return False

    def exit(self):
        """
        Exit the API.

        This method stops all tasks and services, and signals the termination of the API.
        """
        for _task in sorted(
                filter(lambda x: x['status'] and x['on'] == 'shutdown', self.tasks), key=lambda x: x['priority']
        ):
            self.run_task(_task['name'])

        for process in self.processes:
            process.stop()

        self.event.set()

        exit(1)
