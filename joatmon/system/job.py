import datetime
import uuid

from joatmon.core.event import AsyncEvent
from joatmon.core.exception import CoreException
from joatmon.core.utility import new_object_id, to_list_async
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
    status = Field(bool, nullable=False, default=True)
    interval = Field(int, nullable=True)
    script = Field(str, nullable=False)
    arguments = Field(str, nullable=False, default='')
    created_at = Field(datetime.datetime, nullable=False, default=datetime.datetime.now)
    updated_at = Field(datetime.datetime, nullable=False, default=datetime.datetime.now)
    last_run_time = Field(datetime.datetime, nullable=True)
    next_run_time = Field(datetime.datetime, nullable=True)
    is_deleted = Field(bool, nullable=False, default=False)


Job = create_new_type(Job, (Document,))


class JobModule(Module):
    def __init__(self, system):
        super().__init__(system)

        self.events = {
            'on_start': AsyncEvent(),
            'on_end': AsyncEvent(),
            'on_error': AsyncEvent()
        }

    async def _on_start(self, job):
        print(f'job {job} started')

    async def _on_end(self, job):
        print(f'job {job} ended')

    async def _on_error(self, job):
        print(f'job {job} ended with error')

    async def create(self, name, description, priority, status, interval, script: str, arguments):
        await self.system.persistence.drop(Job)

        script = self.system.file_system._get_host_path(script)

        if not self.system.file_system.exists(script):
            raise JobException(f'{self.system.file_system._get_system_path(script)} does not exist')
        if not self.system.file_system.isfile(script):
            raise JobException(f'{self.system.file_system._get_system_path(script)} is not a file')

        if interval is None or interval <= 0:
            interval = 60 * 60 * 24
        else:
            interval = None

        await self.system.persistence.insert(
            Job, {
                'name': name,
                'description': description,
                'priority': priority,
                'status': status,
                'interval': interval,
                'script': script,
                'arguments': arguments,
            }
        )

    async def run(self, object_id):
        ...

    async def stop(self, object_id):
        ...

    async def list(self):
        ret = await to_list_async(self.system.persistence.read(Job, {'is_deleted': False}))
        print(ret)

    async def get(self, object_id):
        ...

    async def remove(self, object_id):
        ...

    async def update(self, object_id):
        ...

    async def start(self):
        self.events['on_start'] += self._on_start
        self.events['on_end'] += self._on_end
        self.events['on_error'] += self._on_error

        await self.list()

    async def shutdown(self):
        self.events['on_start'] -= self._on_start
        self.events['on_end'] -= self._on_end
        self.events['on_error'] -= self._on_error
