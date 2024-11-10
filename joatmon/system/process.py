import asyncio
import datetime
import os
import subprocess
import sys
import time
import typing
import uuid
from enum import auto

from joatmon.core.event import AsyncEvent
from joatmon.core.exception import CoreException
from joatmon.core.utility import first_async, new_object_id, to_list_async
from joatmon.orm import enum
from joatmon.orm.document import Document, create_new_type
from joatmon.orm.field import Field
from joatmon.orm.meta import Meta
from joatmon.system.job import Job
from joatmon.system.module import Module
from joatmon.system.service import Service
from joatmon.system.task import Task


class ProcessType(enum.Enum):
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
    started_at = Field(datetime.datetime, nullable=False, default=datetime.datetime.now)


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

    async def _on_start(self, obj):
        process = subprocess.Popen([sys.executable, obj.script] + obj.arguments.split(' '), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._processes.append((obj, process))

        await self.system.persistence.insert(
            Process, {
                'pid': process.pid,
                'type': int(ProcessType.TASK) if isinstance(obj, Task) else int(ProcessType.JOB) if isinstance(obj, Job) else int(ProcessType.SERVICE),
                'info_id': obj.id,
                'started_at': datetime.datetime.now()
            }
        )

        if isinstance(obj, Task):
            await self.system.task_manager.events['on_start'].fire(obj)
        if isinstance(obj, Job):
            await self.system.job_manager.events['on_start'].fire(obj)
        if isinstance(obj, Service):
            await self.system.service_manager.events['on_start'].fire(obj)

    async def _on_end(self, obj):
        await self.system.persistence.delete(Process, {'info_id': obj.id})

        if isinstance(obj, Task):
            await self.system.task_manager.events['on_end'].fire(obj)
        if isinstance(obj, Job):
            await self.system.job_manager.events['on_end'].fire(obj)
        if isinstance(obj, Service):
            await self.system.service_manager.events['on_end'].fire(obj)

    async def _on_error(self, obj):
        await self.system.persistence.delete(Process, {'info_id': obj.id})

        if isinstance(obj, Task):
            await self.system.task_manager.events['on_error'].fire(obj)
        if isinstance(obj, Job):
            await self.system.job_manager.events['on_error'].fire(obj)
        if isinstance(obj, Service):
            await self.system.service_manager.events['on_error'].fire(obj)

    async def _runner_loop(self):
        while True:
            ended_processes = list(filter(lambda x: x[1].returncode is not None, self._processes))
            for info, process in ended_processes:
                if process.returncode == 0:
                    await self.events['on_end'].fire(info)
                if process.returncode != 0:
                    await self.events['on_error'].fire(info)

            self._processes = list(filter(lambda x: x[1].returncode is None, self._processes))
            for info, process in self._processes:
                ret = process.poll()
                if ret is None:
                    continue

                # for c in iter(process.stdout.read, b''):
                #     print(c)
                # for c in iter(process.stderr.read, b''):
                #     print(c)

            time.sleep(0.1)

    async def run(self, obj: typing.Union[Task, Job, Service]):
        process = await first_async(self.system.persistence.read(Process, {'info_id': obj.id}))
        if process:
            raise ProcessException(f'{obj} is already running')

        await self.events['on_start'].fire(obj)

    async def stop(self, object_id):
        ...

    async def list(self):
        return await to_list_async(self.system.persistence.read(Process, {}))

    async def get(self, object_id):
        ...

    async def start(self):
        self.events['on_start'] += self._on_start
        self.events['on_end'] += self._on_end
        self.events['on_error'] += self._on_error

        self._runner = asyncio.create_task(self._runner_loop())

    async def shutdown(self):
        processes = await self.list()
        for process in processes:
            os.kill(process.pid, 9)

        time.sleep(1)

        if self._runner and not self._runner.done():
            self._runner.cancel()

        # kill each running process
        # send all events

        self.events['on_start'] -= self._on_start
        self.events['on_end'] -= self._on_end
        self.events['on_error'] -= self._on_error
