import asyncio
import datetime
import os
import typing
import uuid
from enum import auto

from transitions import Machine

from joatmon.core import context
from joatmon.core.exception import CoreException
from joatmon.core.utility import convert_size, new_object_id, to_list_async
from joatmon.orm import enum
from joatmon.orm.document import Document, create_new_type
from joatmon.orm.field import Field
from joatmon.orm.meta import Meta


class Task(Meta):
    __collection__ = 'task'

    structured = True
    force = True

    id = Field(uuid.UUID, nullable=False, default=new_object_id, primary=True)
    name = Field(str, nullable=False, default='')
    description = Field(str, nullable=False, default='')
    priority = Field(int, nullable=False, default=10)
    status = Field(bool, nullable=False, default=True)
    mode = Field(str, nullable=False, default='manual')
    interval = Field(int, nullable=True)
    script = Field(str, nullable=False)
    arguments = Field(str, nullable=False, default='')
    created_at = Field(datetime.datetime, nullable=False, default=datetime.datetime.now)
    updated_at = Field(datetime.datetime, nullable=False, default=datetime.datetime.now)
    last_run_time = Field(datetime.datetime, nullable=True)
    next_run_time = Field(datetime.datetime, nullable=True)


class Service(Meta):
    __collection__ = 'task'

    structured = True
    force = True

    id = Field(uuid.UUID, nullable=False, default=new_object_id, primary=True)
    name = Field(str, nullable=False, default='')
    description = Field(str, nullable=False, default='')
    priority = Field(int, nullable=False, default=10)
    status = Field(bool, nullable=False, default=True)
    mode = Field(str, nullable=False, default='manual')
    retry = Field(int, nullable=True)
    script = Field(str, nullable=False)
    arguments = Field(str, nullable=False, default='')
    created_at = Field(datetime.datetime, nullable=False, default=datetime.datetime.now)
    updated_at = Field(datetime.datetime, nullable=False, default=datetime.datetime.now)
    last_run_time = Field(datetime.datetime, nullable=True)
    next_run_time = Field(datetime.datetime, nullable=True)


class State(enum.Enum):
    RUNNING = auto()
    PAUSED = auto()
    STOPPED = auto()
    COMPLETED = auto()
    FAILED = auto()


class Result(enum.Enum):
    SUCCESS = auto()
    FAILURE = auto()


class OSException(CoreException):
    ...


class Process:
    def __init__(self, info: typing.Union[Task, Service], loop: asyncio.AbstractEventLoop, **kwargs):
        states = ['none', 'started', 'stopped', 'running', 'exception', 'starting', 'stopping']
        self.machine = Machine(model=self, states=states, initial='none')

        self.info = info
        self.loop = loop
        self.kwargs = kwargs
        self.process_id = uuid.uuid4()
        self.event = asyncio.Event()
        self.task = None

    def __str__(self):
        return f'{self.process_id} {self.info} {self.machine.state}'

    def __repr__(self):
        return str(self)

    @staticmethod
    def help():
        """
        Provide help about the Runnable.

        Returns:
            dict: An empty dictionary as this method needs to be implemented in subclasses.
        """
        return {}

    async def run(self):
        """
        Run the task.

        This method needs to be implemented in subclasses.
        """
        raise NotImplementedError

    def running(self):
        """
        Check if the task is running.

        Returns:
            bool: True if the task is running, False otherwise.
        """
        return not self.task.done()

    def start(self):
        """
        Start the task.

        This method starts the task by setting the state to 'starting', firing the 'begin' event, and then setting the state to 'started'.
        """
        self.machine.set_state('starting')
        self.task = asyncio.ensure_future(self.run(), loop=self.loop)
        self.machine.set_state('started')

    def stop(self):
        """
        Stop the task.

        This method stops the task by setting the state to 'stopping', firing the 'end' event, setting the event, setting the state to 'stopped', and then cancelling the task.
        """
        self.machine.set_state('stopping')
        self.event.set()
        if self.task and not self.task.done():
            self.task.cancel()
        self.machine.set_state('stopped')


# Task = create_new_type(Task, (Document,))


class Module:
    def __init__(self, name):
        self.name = name


class FSModule(Module):
    def __init__(self, root):
        super().__init__('fs')

        self._root = root
        self._cwd = '/'

    def _get_host_path(self, path):
        if os.path.isabs(path):
            return os.path.abspath(os.path.join(self._root, path))
        else:
            return os.path.abspath(os.path.join(os.path.abspath(self._root), './' + os.path.abspath(os.path.join(self._cwd, path))))

    def touch(self, file):
        with open(self._get_host_path(file), 'w'):
            pass

    def ls(self, path):
        content = list(os.listdir(self._get_host_path(path)))
        content.sort(key=lambda x: (not x.startswith('.'), not os.path.isdir(os.path.join(self._get_host_path(path), x)), x))

        for x in content:
            if os.path.isdir(os.path.join(self._get_host_path(path), x)):
                size, unit = convert_size(os.path.getsize(os.path.join(self._get_host_path(path), x))).split(' ')
                print(f'{size:>8} {unit:>2} {x}/')
            elif os.path.isfile(x):
                size, unit = convert_size(os.path.getsize(os.path.join(self._get_host_path(path), x))).split(' ')
                print(f'{size:>8} {unit:>2} {x}')

    def cd(self, path):
        if os.path.isabs(path):
            self._cwd = path
        else:
            self._cwd = os.path.abspath(os.path.join(self._cwd, path))

    def mkdir(self):
        ...

    def rm(self):
        ...


class OS:
    def __init__(self, root):
        self.modules = [
            FSModule(root)
        ]

        self.db = context.get_value('sqlite')

    def __getattr__(self, item):
        for module in self.modules:
            if getattr(module, item, None):
                return getattr(module, item)
        else:
            raise OSException(f'{item} is not a valid executable')

    async def create_task(self, name, description, priority, status, mode, interval, script, arguments):
        await self.db.drop(Task)

        if not os.path.isabs(script):
            script = os.path.abspath(os.path.join(self._cwd, script))

        if not os.path.exists(script):
            raise OSException(f'{script} does not exist')

        if mode not in ['manual', 'interval', 'startup', 'shutdown']:
            raise OSException(f'{mode} is not a valid mode')

        if mode in ['interval'] and (interval is None or interval <= 0):
            interval = 60 * 60 * 24
        else:
            interval = None

        await self.db.insert(
            Task, {
                'name': name,
                'description': description,
                'priority': priority,
                'status': status,
                'mode': mode,
                'interval': interval,
                'script': script,
                'arguments': arguments,
            }
        )
        ret = await to_list_async(self.db.read(Task, {}))
        print(ret)
