import asyncio
import datetime
import uuid

from joatmon.core.event import AsyncEvent
from joatmon.core.exception import CoreException
from joatmon.core.utility import first_async, new_object_id, to_list_async
from joatmon.orm.document import Document, create_new_type
from joatmon.orm.field import Field
from joatmon.orm.meta import Meta
from joatmon.system.module import Module


class JobException(CoreException):
    ...


class Job(Meta):
    __collection__ = 'job'

    structured = True
    force = True

    id = Field(uuid.UUID, nullable=False, default=new_object_id, primary=True)
    name = Field(str, nullable=False, default='')
    description = Field(str, nullable=False, default='')
    priority = Field(int, nullable=False, default=10)
    interval = Field(int, nullable=False, default=60 * 60 * 24)
    script = Field(str, nullable=False)
    arguments = Field(str, nullable=False, default='')
    created_at = Field(datetime.datetime, nullable=False, default=datetime.datetime.now)
    updated_at = Field(datetime.datetime, nullable=False, default=datetime.datetime.now)
    last_run_time = Field(datetime.datetime, nullable=True)


Job = create_new_type(Job, (Document,))


class JobModule(Module):
    def __init__(self, system):
        super().__init__(system)

        self.events = {
            'on_start': AsyncEvent(),
            'on_end': AsyncEvent(),
            'on_error': AsyncEvent()
        }

        self._runner = None

    async def _on_start(self, job):
        job.last_run_time = datetime.datetime.now()
        await self.system.persistence.update(Job, {'id': job.id}, job)

    async def _on_end(self, job):
        ...

    async def _on_error(self, job):
        ...

    async def _runner_loop(self):
        while True:
            jobs = await self.list()
            for job in jobs:
                if job.last_run_time and job.last_run_time + datetime.timedelta(seconds=job.interval) > datetime.datetime.now():
                    continue

                await self.run(job.id)

            await asyncio.sleep(0.1)

    async def create(self, name, description, priority, interval, script: str, arguments):
        job = await first_async(self.system.persistence.read(Job, {'name': name}))

        script = self.system.file_system._get_host_path(script)  # full path or task name

        if not self.system.file_system.exists(script):
            raise JobException(f'{self.system.file_system._get_system_path(script)} does not exist')
        if not self.system.file_system.isfile(script):
            raise JobException(f'{self.system.file_system._get_system_path(script)} is not a file')

        if interval is None or interval <= 0:
            interval = 60 * 60 * 24

        if job:
            await self.system.persistence.update(
                Job, {'id': job.id}, {
                    'name': name,
                    'description': description,
                    'priority': priority,
                    'interval': interval,
                    'script': script,
                    'arguments': arguments,
                }
            )
        else:
            await self.system.persistence.insert(
                Job, {
                    'name': name,
                    'description': description,
                    'priority': priority,
                    'interval': interval,
                    'script': script,
                    'arguments': arguments,
                }
            )

    async def run(self, object_id):
        job = await first_async(self.system.persistence.read(Job, {'id': object_id}))
        await self.system.process_manager.run(job)

    async def stop(self, object_id):
        ...

    async def list(self):
        return await to_list_async(self.system.persistence.read(Job, {}))

    async def get(self, object_id):
        return await first_async(self.system.persistence.read(Job, {'id': object_id}))

    async def remove(self, object_id):
        await self.system.persistence.delete(Job, {'id': object_id})

    async def update(self, object_id):
        ...

    async def start(self):
        self.events['on_start'] += self._on_start
        self.events['on_end'] += self._on_end
        self.events['on_error'] += self._on_error

        self._runner = asyncio.create_task(self._runner_loop())

    async def shutdown(self):
        if self._runner and not self._runner.done():
            self._runner.cancel()

        self.events['on_start'] -= self._on_start
        self.events['on_end'] -= self._on_end
        self.events['on_error'] -= self._on_error
