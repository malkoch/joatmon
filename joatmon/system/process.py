import asyncio
import typing
import uuid
from enum import auto

from transitions import Machine

from joatmon.orm import enum
from joatmon.system.job import Job
from joatmon.system.module import Module
from joatmon.system.service import Service
from joatmon.system.task import Task


class State(enum.Enum):
    RUNNING = auto()
    PAUSED = auto()
    STOPPED = auto()
    COMPLETED = auto()
    FAILED = auto()


class Result(enum.Enum):
    SUCCESS = auto()
    FAILURE = auto()


class Process:
    def __init__(self, info: typing.Union[Task, Job, Service], loop: asyncio.AbstractEventLoop, **kwargs):
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


class ProcessModule(Module):
    def __init__(self, system):
        super().__init__(system)

    def create(self):
        ...

    def start(self):
        ...

    def stop(self):
        ...

    def list(self):
        ...

    def get(self):
        ...

    def remove(self):
        ...

    def update(self):
        ...

    def run(self):
        ...

    def shutdown(self):
        ...
