import datetime
import uuid

from joatmon.core.event import AsyncEvent
from joatmon.core.exception import CoreException
from joatmon.core.utility import first_async, new_object_id, to_list_async
from joatmon.orm.document import Document, create_new_type
from joatmon.orm.field import Field
from joatmon.orm.meta import Meta
from joatmon.system.module import Module


class TaskException(CoreException):
    ...


class Task(Meta):
    __collection__ = 'task'

    structured = True
    force = True

    id = Field(uuid.UUID, nullable=False, default=new_object_id, primary=True)
    name = Field(str, nullable=False, default='')
    description = Field(str, nullable=False, default='')
    priority = Field(int, nullable=False, default=10)
    mode = Field(str, nullable=False, default='manual')
    script = Field(str, nullable=False)
    arguments = Field(str, nullable=False, default='')
    created_at = Field(datetime.datetime, nullable=False, default=datetime.datetime.now)
    updated_at = Field(datetime.datetime, nullable=False, default=datetime.datetime.now)


Task = create_new_type(Task, (Document,))


class TaskModule(Module):
    def __init__(self, system):
        super().__init__(system)

        self.events = {
            'on_start': AsyncEvent(),
            'on_end': AsyncEvent(),
            'on_error': AsyncEvent()
        }

    async def _on_start(self, task):
        ...

    async def _on_end(self, task):
        ...

    async def _on_error(self, task):
        ...

    async def create(self, name, description, priority, mode, script: str, arguments):
        task = await first_async(self.system.persistence.read(Task, {'name': name}))

        script = self.system.file_system._get_host_path(script)

        if not self.system.file_system.exists(script):
            raise TaskException(f'{self.system.file_system._get_system_path(script)} does not exist')
        if not self.system.file_system.isfile(script):
            raise TaskException(f'{self.system.file_system._get_system_path(script)} is not a file')

        if mode not in ['manual', 'startup', 'shutdown']:
            raise TaskException(f'{mode} is not a valid mode')

        if task:
            await self.system.persistence.update(
                Task, {'id': task.id}, {
                    'name': name,
                    'description': description,
                    'priority': priority,
                    'mode': mode,
                    'script': script,
                    'arguments': arguments,
                }
            )
        else:
            await self.system.persistence.insert(
                Task, {
                    'name': name,
                    'description': description,
                    'priority': priority,
                    'mode': mode,
                    'script': script,
                    'arguments': arguments,
                }
            )

    async def run(self, object_id):
        task = await first_async(self.system.persistence.read(Task, {'id': object_id}))
        await self.system.process_manager.run(task)

    async def stop(self, object_id):
        ...

    async def list(self):
        return await to_list_async(self.system.persistence.read(Task, {}))

    async def get(self, object_id):
        return await first_async(self.system.persistence.read(Task, {'id': object_id}))

    async def remove(self, object_id):
        await self.system.persistence.delete(Task, {'id': object_id})

    async def update(self, object_id):
        ...

    async def start(self):
        self.events['on_start'] += self._on_start
        self.events['on_end'] += self._on_end
        self.events['on_error'] += self._on_error

        tasks = await self.list()
        for task in filter(lambda x: x.mode == 'startup', tasks):
            await self.system.process_manager.run(task)

    async def shutdown(self):
        tasks = await self.list()
        for task in filter(lambda x: x.mode == 'shutdown', tasks):
            await self.system.process_manager.run(task)

        self.events['on_start'] -= self._on_start
        self.events['on_end'] -= self._on_end
        self.events['on_error'] -= self._on_error
