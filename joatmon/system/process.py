import asyncio
import datetime
import subprocess
import sys
import time
import typing
import uuid
from enum import auto

from joatmon.core.event import AsyncEvent
from joatmon.core.exception import CoreException
from joatmon.core.utility import new_object_id
from joatmon.orm import enum
from joatmon.orm.document import Document, create_new_type
from joatmon.orm.field import Field
from joatmon.orm.meta import Meta
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


class ProcessType:
    TASK = auto()
    JOB = auto()
    SERVICE = auto()


class ProcessException(CoreException):
    ...


class Process(Meta):
    __collection__ = 'process'

    structured = True
    force = True

    id = Field(uuid.UUID, nullable=False, default=new_object_id, primary=True)
    pid = Field(int, nullable=False)
    type = Field(int, nullable=False, default=int(ProcessType.TASK))
    info_id = Field(uuid.UUID, nullable=False)
    state = Field(int, nullable=False, default=int(State.RUNNING))
    started_at = Field(datetime.datetime, nullable=False, default=datetime.datetime.now)
    ended_at = Field(datetime.datetime, nullable=True)
    end_code = Field(int, nullable=True)
    end_reason = Field(str, nullable=True)
    is_deleted = Field(bool, nullable=False, default=False)


Process = create_new_type(Process, (Document,))


class ProcessModule(Module):
    def __init__(self, system):
        super().__init__(system)

        self.events = {
            'on_start': AsyncEvent(),
            'on_end': AsyncEvent(),
            'on_error': AsyncEvent()
        }

        self._processes = []
        self._runner = None

    async def _on_start(self, process):
        if isinstance(process, Task):
            await self.system.task_manager.events['on_start'].fire(process)
        if isinstance(process, Job):
            await self.system.job_manager.events['on_start'].fire(process)
        if isinstance(process, Service):
            await self.system.service_manager.events['on_start'].fire(process)

    async def _on_end(self, process):
        if isinstance(process, Task):
            await self.system.task_manager.events['on_end'].fire(process)
        if isinstance(process, Job):
            await self.system.job_manager.events['on_end'].fire(process)
        if isinstance(process, Service):
            await self.system.service_manager.events['on_end'].fire(process)

    async def _on_error(self, process):
        if isinstance(process, Task):
            await self.system.task_manager.events['on_error'].fire(process)
        if isinstance(process, Job):
            await self.system.job_manager.events['on_error'].fire(process)
        if isinstance(process, Service):
            await self.system.service_manager.events['on_error'].fire(process)

    async def _runner_loop(self):
        while self._alive:
            ended_processes = list(filter(lambda x: x[1].returncode is not None, self._processes))
            for ended_process in ended_processes:
                if ended_process[1].returncode == 0:
                    await self.events['on_end'].fire(ended_process[0])
                if ended_process[1].returncode != 0:
                    await self.events['on_error'].fire(ended_process[0])

            self._processes = list(filter(lambda x: x[1].returncode is None, self._processes))
            for info, process in self._processes:
                ret = process.poll()
                if ret is None:
                    continue

                for line in iter(process.stdout.read, b''):
                    print(line)
                for line in iter(process.stderr.read, b''):
                    print(line)

            time.sleep(0.1)

    async def run(self, obj: typing.Union[Task, Job, Service]):
        process = subprocess.Popen([sys.executable, obj.script] + obj.arguments.split(' '), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        await self.events['on_start'].fire(obj)
        self._processes.append((obj, process))

    async def stop(self, object_id):
        ...

    async def list(self):
        ...

    async def get(self, object_id):
        ...

    async def start(self):
        self._alive = True

        self.events['on_start'] += self._on_start
        self.events['on_end'] += self._on_end
        self.events['on_error'] += self._on_error

        self._runner = asyncio.create_task(self._runner_loop())

    async def shutdown(self):
        self._alive = False

        if self._runner and not self._runner.done():
            self._runner.cancel()

        # kill each running process
        # send all events

        self.events['on_start'] -= self._on_start
        self.events['on_end'] -= self._on_end
        self.events['on_error'] -= self._on_error
